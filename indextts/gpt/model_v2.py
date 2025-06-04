from dataclasses import dataclass
import numpy as np
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
import torch
import torch.nn.functional as F

# from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessorList
# from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from mlx_lm.models.gpt2 import ModelArgs as GPT2Config, TransformerBlock
from mlx_lm.generate import wired_limit, generation_stream
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx.utils import tree_flatten, tree_unflatten
import mlx.core as mx
import mlx.nn as nn
from .conformer_encoder import ConformerEncoder
from .perceiver import PerceiverResampler


class NullEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, x: mx.array):
        if x.ndim == 1:
            x = x.reshape((1, x.shape[0]))
        return mx.zeros((x.shape[0], x.shape[1], self.dim), dtype=mx.float32)

@dataclass
class ModelArgs(GPT2Config):
    start_mel_token: int = 8192
    stop_mel_token: int = 8193
    start_text_token: int = 0
    stop_text_token: int = 1

class GPT2ForMel(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
    ):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_positions = config.n_positions
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.ln_f = nn.LayerNorm(self.n_embd, eps=self.layer_norm_epsilon)
        self.h = [TransformerBlock(args=config) for _ in range(self.n_layer)]
        self._final_norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self._mel_head = nn.Linear(config.n_embd, config.vocab_size)
        self.config = config

    def store_mel_emb(self, cache_emb, mask:mx.array=None):
        self._cache_emb = cache_emb
        self._cache_mask = mask if not mask.all() else None

    def _create_4d_causal_mask(self, b, L, past_kv_len=0):
        # If no mask or mask is all True, return "causal" or None depending on past_kv_len
        if self._cache_mask is None:
            return "causal" if past_kv_len == 0 else None

        offset = self._cache_emb.shape[1]
        mask = mx.pad(self._cache_mask, [(0, 0), (0, past_kv_len + L - offset)], constant_values=1)
        mask = (mask == 1)[:, None, None, :]  # [b, 1, 1, L]

        if past_kv_len == 0:
            # Build full causal mask
            rinds = mx.arange(L)
            linds = rinds[:, None]
            rinds = rinds[None]
            causal_mask = linds >= rinds  # triangular mask
            causal_mask = mx.broadcast_to(causal_mask[None, None, :, :], (b, 1, L, L))
            can_attend = mask.transpose(0, 1, 3, 2)  # [b, 1, L, 1]
            mask = causal_mask & mask & can_attend
        return mask

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
        output_hidden_states=False,
    ):
        # print("x", inputs.shape)
        b, L = inputs.shape
        past_len = 0
        if cache is not None and len(cache) > 0 and cache[0] is not None:
            past_len = cache[0].offset

        mel_pos_offset = self._cache_emb.shape[1]
        if past_len == 0:
            # first time we call the model
            if L - mel_pos_offset > 0:
                inputs_embeds = self._mel_embeddings(inputs[:, mel_pos_offset:]) + self._mel_pos_embedding(
                    mx.arange(0, L - mel_pos_offset, dtype=mx.int32)
                )
                inputs_embeds = mx.concat([self._cache_emb, inputs_embeds], axis=1)
            else:
                inputs_embeds = self._cache_emb
            # (b, 1, L, L) or "causal"
            mask = self._create_4d_causal_mask(b, L)
        else:
            assert past_len >= self._cache_emb.shape[1], (
                f"cache: {cache[0]}, L: {L}, past_len: {past_len}, emb: {self._cache_emb.shape[1]}"
            )
            cur_pos = past_len - mel_pos_offset
            cur_pos = mx.arange(cur_pos, cur_pos + L, dtype=mx.int32)
            inputs_embeds = self._mel_embeddings(inputs) + self._mel_pos_embedding(cur_pos)
            # (b, 1, 1, L)
            mask = self._create_4d_causal_mask(b, L, past_len)
        # if mask is not None and isinstance(mask, mx.array):
            # mask = mx.where(~mask, mx.finfo(inputs_embeds.dtype).min, mx.array(0, dtype=inputs_embeds.dtype))
        hidden_states = inputs_embeds
        if cache is None:
            cache = [None] * len(self.h)

        for block, c in zip(self.h, cache):
            hidden_states = block(hidden_states, mask, cache=c)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = self._final_norm(hidden_states)
        # print("out:", hidden_states.shape)
        if output_hidden_states:
            return hidden_states
        mel_logits = self._mel_head(hidden_states)
        return mel_logits

    @property
    def layers(self):
        return self.h

    def tie_custom_weights(
        self,
        mel_embeddings: nn.Embedding,
        mel_pos_embedding: nn.Embedding,
        text_embeddings: nn.Embedding,
        text_pos_embedding: nn.Embedding,
        final_norm: nn.LayerNorm,
        mel_head: nn.Linear,
    ):
        self._mel_embeddings = mel_embeddings
        self._mel_pos_embedding = mel_pos_embedding
        self._text_embeddings = text_embeddings
        self._text_pos_embedding = text_pos_embedding
        self._final_norm.weight = final_norm.weight
        self._final_norm.bias = final_norm.bias
        self._mel_head.weight = mel_head.weight
        self._mel_head.bias = mel_head.bias


    def sanitize(self, weights: Dict[str, mx.array]):
        for i in range(self.n_layer):
            if f"h.{i}.attn.bias" in weights:
                del weights[f"h.{i}.attn.bias"]
            if f"h.{i}.attn.c_attn.weight" in weights:
                weights[f"h.{i}.attn.c_attn.weight"] = weights[f"h.{i}.attn.c_attn.weight"].transpose(1, 0)
            if f"h.{i}.attn.c_proj.weight" in weights:
                weights[f"h.{i}.attn.c_proj.weight"] = weights[f"h.{i}.attn.c_proj.weight"].transpose(1, 0)
            if f"h.{i}.mlp.c_fc.weight" in weights:
                weights[f"h.{i}.mlp.c_fc.weight"] = weights[f"h.{i}.mlp.c_fc.weight"].transpose(1, 0)
            if f"h.{i}.mlp.c_proj.weight" in weights:
                weights[f"h.{i}.mlp.c_proj.weight"] = weights[f"h.{i}.mlp.c_proj.weight"].transpose(1, 0)
        
        return weights

class UnifiedVoiceMLX(nn.Module):
    def __init__(
        self,
        layers=8,
        model_dim=512,
        heads=8,
        max_text_tokens=120,
        max_mel_tokens=250,
        max_conditioning_inputs=1,
        mel_length_compression=1024,
        number_text_tokens=256,
        start_text_token=0,
        stop_text_token=1,
        number_mel_codes=8194,
        start_mel_token=8192,
        stop_mel_token=8193,
        train_solo_embeddings=False,
        use_mel_codes_as_input=True,
        checkpointing=True,
        types=1,
        activation_function=None,
        condition_num_latent=32,
        condition_type="conformer_perceiver",
        condition_module=None,
    ):
        """
        Args:
            layers: Number of layers in transformer stack.
            model_dim: Operating dimensions of the transformer
            heads: Number of transformer heads. Must be divisible by model_dim. Recommend model_dim//64
            max_text_tokens: Maximum number of text tokens that will be encountered by model.
            max_mel_tokens: Maximum number of MEL tokens that will be encountered by model.
            max_conditioning_inputs: Maximum number of conditioning inputs provided to the model. If (1), conditioning input can be of format (b,80,s), otherwise (b,n,80,s).
            mel_length_compression: The factor between <number_input_samples> and <mel_tokens>. Used to compute MEL code padding given wav input length.
            number_text_tokens:
            start_text_token:
            stop_text_token:
            number_mel_codes:
            start_mel_token:
            stop_mel_token:
            train_solo_embeddings:
            use_mel_codes_as_input:
            checkpointing:
            condition_type: perceiver, gst or default encoder
        """
        super().__init__()
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.layers = layers
        self.heads = heads
        self.max_mel_tokens = max_mel_tokens
        self.max_text_tokens = max_text_tokens
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.mel_length_compression = mel_length_compression
        self.condition_type = condition_type
        self.cond_num = condition_num_latent
        self.cond_mask_pad = torch.nn.ConstantPad1d((self.cond_num, 0), True)

        assert condition_type == "conformer_perceiver"
        self.conditioning_encoder = ConformerEncoder(
            input_size=100,
            output_size=condition_module["output_size"],
            linear_units=condition_module["linear_units"],
            attention_heads=condition_module["attention_heads"],
            num_blocks=condition_module["num_blocks"],
            input_layer=condition_module["input_layer"],
        )

        self.perceiver_encoder = PerceiverResampler(
            model_dim,
            dim_context=condition_module["output_size"],
            ff_mult=condition_module["perceiver_mult"],
            heads=condition_module["attention_heads"],
            num_latents=self.cond_num,
        )

        self.text_embedding = nn.Embedding(self.number_text_tokens * types + 1, model_dim)
        assert use_mel_codes_as_input, "Mel codes as input is only supported for this model."
        self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)
        n_mel_pos = self.max_mel_tokens + 2 + self.max_conditioning_inputs
        n_text_pos = self.max_text_tokens + 2
        self.mel_pos_embedding = nn.Embedding(n_mel_pos, model_dim)
        self.text_pos_embedding = nn.Embedding(n_text_pos, model_dim)

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens * types + 1)
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes)

        seq_length = self.cond_num + n_text_pos + n_mel_pos
        gpt_config = ModelArgs(
            model_type="gpt2",
            vocab_size=self.number_mel_codes,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=self.model_dim,
            n_layer=self.layers,
            n_head=self.heads,
            layer_norm_epsilon=1e-5,
            start_mel_token=self.start_mel_token,
            stop_mel_token=self.stop_mel_token,
            start_text_token=self.start_text_token,
            stop_text_token=self.stop_text_token,
        )
        self.gpt = GPT2ForMel(gpt_config)

    def to(self, device, *args, **kwargs):
        # move torch modules to device
        self.conditioning_encoder.to(device)
        self.perceiver_encoder.to(device)
        return self

    def eval(self):
        super().eval()
        self.conditioning_encoder.eval()
        self.perceiver_encoder.eval()
        return self
    def half(self):
        self.conditioning_encoder.half()
        self.perceiver_encoder.half()
        self.set_dtype(mx.float16)
        mx.eval(self.parameters())
        return self
    def load_state_dict(self, state_dict: Dict[str, Any], *args, **kwargs):
        """Load pytorch state dict into this mlx model."""
        gpt_weights = {}
        self_weights = {}
        conditioning_encoder_weights = {}
        perceiver_encoder_weights = {}
        for key in list(state_dict.keys()):
            if key.startswith("gpt."):
                new_key = key.replace("gpt.", "")
                gpt_weights[new_key] = mx.array(state_dict[key]).astype(mx.float32)
                # del state_dict[key]
            elif key.startswith(
                (
                    "mel_pos_embedding.",
                    "mel_embedding.",
                    "text_pos_embedding.",
                    "text_embedding.",
                    "final_norm.",
                    "text_head.",
                    "mel_head.",
                )
            ):
                if key.startswith(("mel_pos_embedding.emb.", "text_pos_embedding.emb.")):
                    new_key = key.replace("emb.", "")
                else:
                    new_key = key
                # linear weights
                self_weights[new_key] = mx.array(state_dict[key]).astype(mx.float32)
            # remaining torch weights
            elif key.startswith("conditioning_encoder."):
                new_key = key.replace("conditioning_encoder.", "")
                conditioning_encoder_weights[new_key] = state_dict[key]
            elif key.startswith("perceiver_encoder."):
                new_key = key.replace("perceiver_encoder.", "")
                perceiver_encoder_weights[new_key] = state_dict[key]
            else:
                print("unknown", key, "dtype", state_dict[key].dtype, "shape", state_dict[key].shape)
        gpt_weights = self.gpt.sanitize(gpt_weights)
        params = dict(tree_flatten(self.gpt.parameters()))
        # print(params.keys())
        for k in params.keys():
            if k not in gpt_weights:
                print("missing", k)
            elif params[k].shape != gpt_weights[k].shape:
                print(k, "expected", params[k].shape, "got", gpt_weights[k].shape)
        self.gpt.update(tree_unflatten([(k, v) for k, v in gpt_weights.items()]))
        params = dict(tree_flatten(self.parameters()))
        for k in self_weights:
            # print("self_weights", k, self_weights[k].shape)
            if k not in params:
                print("missing", k)
            elif params[k].shape != self_weights[k].shape:
                print(k, "expected", params[k].shape, "got", self_weights[k].shape)
        self.update(tree_unflatten([(k, v) for k, v in self_weights.items()]))
        self.conditioning_encoder.load_state_dict(conditioning_encoder_weights)
        if len(perceiver_encoder_weights) > 0:
            self.perceiver_encoder.load_state_dict(perceiver_encoder_weights)

    def post_init_gpt2_config(self, *, half=False, **kwargs):
        self.gpt.tie_custom_weights(
            self.mel_embedding,
            self.mel_pos_embedding,
            self.text_embedding,
            self.text_pos_embedding,
            self.final_norm,
            self.mel_head,
        )
        self.conditioning_encoder.eval()
        self.perceiver_encoder.eval()
        self.eval()
        mx.eval(self.parameters())

    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_mel_padding(self, mel_input_tokens, mel_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(mel_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = mel_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens

    def set_text_padding(self, text_input_tokens, text_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(text_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = text_lengths[b]
            if actual_end < text_input_tokens.shape[-1]:
                text_input_tokens[b, actual_end:] = self.stop_text_token
        return text_input_tokens


    def get_conditioning(self, speech_conditioning_input: torch.Tensor, cond_mel_lengths: torch.Tensor = None):
        assert self.condition_type == "conformer_perceiver"
        # (b, s, d), (b, 1, s)
        speech_conditioning_input, mask = self.conditioning_encoder.forward(
            speech_conditioning_input.transpose(1, 2), cond_mel_lengths
        )  
        conds_mask = self.cond_mask_pad(mask.squeeze(1))
        conds = self.perceiver_encoder(speech_conditioning_input, conds_mask)  # (b, 32, d)
        return conds

    def __call__(
        self,
        speech_conditioning_input: torch.FloatTensor,
        text_inputs: torch.LongTensor,
        text_lengths: torch.LongTensor,
        mel_codes : torch.LongTensor,
        wav_lengths: torch.LongTensor,
        cond_mel_lengths=None,
        types=None,
        return_latent=True,
        speech_conditioning_latent=None,
        **kwargs,
    ):

        if speech_conditioning_latent is None:
            speech_conditioning_latent = self.get_conditioning(speech_conditioning_input, cond_mel_lengths)
        # Types are expressed by expanding the text embedding space.
        if types is not None:
            text_inputs = text_inputs * (1 + types).unsqueeze(-1)

        # Set padding areas within MEL (currently it is coded with the MEL code for <zero>).
        # mel_codes_lengths = torch.div(wav_lengths, self.mel_length_compression, rounding_mode='trunc')
        mel_codes_lengths = torch.ceil(wav_lengths / self.mel_length_compression).long() + 1
        mel_codes = self.set_mel_padding(mel_codes, mel_codes_lengths)
        mel_targets = F.pad(mel_codes, (0, 2), value=self.stop_mel_token)
        text_inputs = self.set_text_padding(text_inputs, text_lengths)
        # text_inputs = F.pad(text_inputs, (1, 0), value=self.start_text_token)
        text_targets = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        # if (mel_codes[:, 0] != self.start_mel_token).any():
        mel_codes = F.pad(mel_codes, (1, 0), value=self.start_mel_token)
        mel_codes = F.pad(mel_codes, (0, 1), value=self.stop_mel_token)
        
        input_ids, inputs_embeds, attention_mask = self.prepare_gpt_inputs(speech_conditioning_latent, text_inputs)
        mel_codes_array = mx.array(mel_codes.to("cpu"))
        mel_emb = self.mel_embedding(mel_codes_array)
        mel_emb = mel_emb + self.mel_pos_embedding(mel_codes_array)
        inputs_embeds = mx.concat([inputs_embeds[:, :-1], mel_emb], axis=1)
        attention_mask = mx.pad(attention_mask[:, -1], (0, mel_codes_array.shape[1]), constant_values=1)
        input_ids = mx.concat([input_ids[:, :-1], mel_codes_array], axis=1)
        self.gpt.store_mel_emb(inputs_embeds, mask=attention_mask)
        gpt_out = self.gpt(
            inputs=input_ids,
            mask="causal",
            cache=None,
            output_hidden_states=True,
        )
        offset = speech_conditioning_latent.shape[1]
        hidden_states = gpt_out[:, offset:]
        hidden_states = self.final_norm(hidden_states)

        if return_latent:
            # Strip off the two tokens added by this forward pass.
            hidden_states = hidden_states[:, -mel_codes.shape[1] : -2]
            if hidden_states.dtype != mx.float32:
                hidden_states = hidden_states.astype(mx.float32)
            mx.eval(hidden_states)
            return torch.tensor(np.array(hidden_states, copy=False), device=text_inputs.device)

        text_logits = self.text_head(hidden_states[:, :text_inputs.shape[1]+1])
        text_logits = text_logits.swapaxes(1, 2)
        mel_logits = self.mel_head(hidden_states[:, -mel_codes.shape[1]:])
        mel_logits = mel_logits.swapaxes(1, 2)
        mx.eval(text_logits, mel_logits)
        text_logits = torch.tensor(text_logits, device=text_inputs.device)
        mel_logits = torch.tensor(mel_logits, device=text_inputs.device)
        loss_text = F.cross_entropy(text_logits, text_targets.long())
        loss_mel = F.cross_entropy(mel_logits, mel_targets.long(), ignore_index=self.stop_mel_token)
        return loss_text.mean(), loss_mel.mean(), mel_logits

    def prepare_gpt_inputs(
        self,
        conditional_latents: torch.FloatTensor,
        text_inputs: torch.Tensor,
    ):
        """
        Prepare the inputs for the GPT2ForMel to generate.
        Args:
            conds_latent: (b, 32, dim) audio conditioning embedding by `get_conditioning()`
            text_inputs: (b, L)
        Returns:
            input_ids: (b, s+1) the input ids
            inputs_embeds: (b, s+1, dim) the input embeddings
            attention_mask: (b, s+1) the attention mask
        """
        b, L = text_inputs.shape[:2]
        conditional_latents = conditional_latents.cpu()
        latent_dtype = conditional_latents.dtype
        text_emb_dtype = self.text_embedding.weight.dtype
        if latent_dtype.itemsize != text_emb_dtype.size:
            print("converting conditional latents to text embedding dtype", latent_dtype, text_emb_dtype)
            if text_emb_dtype == mx.float16:
                conditional_latents = conditional_latents.to(dtype=torch.float16)
            elif text_emb_dtype == mx.float32:
                conditional_latents = conditional_latents.to(dtype=torch.float32)
            elif text_emb_dtype == mx.bfloat16:
                conditional_latents = conditional_latents.to(dtype=torch.bfloat16)
        single_cond = conditional_latents.ndim == 3 and conditional_latents.shape[0] == 1
        if not single_cond:
            assert conditional_latents.shape[0] == b, f"batch size mismatch: {conditional_latents.shape[0]} vs {b}"
        batched_mel_emb = []
        attention_masks = []
        target_len = conditional_latents.shape[1] + L + 2
        start_mel_token = mx.array([self.start_mel_token], dtype=mx.int32)
        # [1, dim]
        start_mel_emb = self.mel_embedding(start_mel_token) + self.mel_pos_embedding(
            mx.zeros((1,), dtype=mx.int32)
        )
        for i in range(b):
            valid_mask = (text_inputs[i] != self.stop_text_token) & (text_inputs[i] != self.start_text_token)
            text_input = text_inputs[i][valid_mask]
            text_input = mx.array(text_input.cpu(), dtype=mx.int32)
            text_input = mx.pad(text_input, (1, 0), constant_values=self.start_text_token)
            text_input = mx.pad(text_input, (0, 1), constant_values=self.stop_text_token)
            text_input_pos = mx.arange(0, text_input.size, dtype=mx.int32)
            text_emb = self.text_embedding(text_input) + self.text_pos_embedding(text_input_pos)
            # concatenate [conditional latents][text embeddings]
            conds_text_emb = [
                mx.array(conditional_latents.squeeze(0) if single_cond else conditional_latents[i]),
                text_emb,
                start_mel_emb,
            ]
            # +1 for the start_mel_token
            attention_mask = mx.ones(target_len + 1, dtype=mx.int32)
            # check this text input is padded
            padding: int = L + 2 - text_input.size
            # pad left of [cond][text] -> [pad][cond][text]
            if padding > 0:
                pad = mx.zeros(
                    (padding, conditional_latents.size(-1)), dtype=text_emb.dtype
                )  # [p, dim]
                conds_text_emb.insert(0, pad)
                attention_mask[:padding] = 0
            mel_emb = mx.concatenate(conds_text_emb, axis=0)  # [s+1, dim]
            assert mel_emb.shape[0] == target_len + 1, f"mel_emb.shape: {mel_emb.shape}, target_len: {target_len}"
            batched_mel_emb.append(mel_emb)
            attention_masks.append(attention_mask)
        # [b, s+1, dim]
        batched_mel_emb = mx.stack(batched_mel_emb, axis=0)
        # [b, s+1]
        attention_mask = mx.stack(attention_masks, axis=0)
        # [b, s+1]
        fake_inputs = mx.ones(
            (
                batched_mel_emb.shape[0],
                batched_mel_emb.shape[1],
            ),
            dtype=mx.int32,
        )
        fake_inputs[:, -1] = self.start_mel_token
        return fake_inputs, batched_mel_emb, attention_mask

    def inference_speech(
        self,
        speech_conditioning_mel: torch.FloatTensor,
        text_inputs: torch.LongTensor,
        cond_mel_lengths=None,
        input_tokens=None,
        speech_conditioning_latent=None,
        num_return_sequences=1,
        max_generate_length=None,
        typical_sampling=False,
        typical_mass=0.9,
        **hf_generate_kwargs,
    ):
        """
        Args:
            speech_conditioning_mel: (b, n_mels, frames) or (n_mels, frames)
            text_inputs: (b, L)
            cond_mel_lengths: lengths of the conditioning mel spectrograms in shape (b,) or (1,)
            input_tokens: additional tokens for generation in shape (b, s) or (1,)
            speech_conditioning_latent: (b, n_latents, dim) audio conditioning embedding by `get_conditioning()`.
                If not None, the `speech_conditioning_mel` and `cond_mel_lengths` will be ignored
            max_generate_length: limit the number of generated tokens
            hf_generate_kwargs: kwargs for `GPT2InferenceModel.generate(**hf_generate_kwargs)`
        """
        if speech_conditioning_latent is None:
            if speech_conditioning_mel.ndim == 2:
                speech_conditioning_mel = speech_conditioning_mel.unsqueeze(0)
            if cond_mel_lengths is None:
                cond_mel_lengths = torch.tensor(
                    [speech_conditioning_mel.shape[-1]], device=speech_conditioning_mel.device
                )
            speech_conditioning_latent = self.get_conditioning(speech_conditioning_mel, cond_mel_lengths)
        logits_processors = make_logits_processors(
            hf_generate_kwargs.get("logit_bias", None),
            hf_generate_kwargs.get("repetition_penalty", 2.0),
            hf_generate_kwargs.get("repetition_context_size", 30),
        )
        sampler = make_sampler(
            1.0,
            top_p=0.95,
            top_k=30,
            # min_p=hf_generate_kwargs.get("min_p") or 0.1,
            # min_tokens_to_keep=hf_generate_kwargs.get("min_tokens_to_keep") or 10,
        ) if hf_generate_kwargs.get("do_sample", True) else None
        batchsize = text_inputs.shape[0]
        output = []
        with wired_limit(self.gpt, [generation_stream]):
            batch_inputs = [
                self.prepare_gpt_inputs(speech_conditioning_latent, text_inputs[b].unsqueeze(0))
                for b in range(batchsize)
            ]
            mx.async_eval(batch_inputs)
            for inputs, embs, mask in batch_inputs:
                trunc_index = inputs.shape[1]
                tokens = inputs.reshape((trunc_index,))
                max_length = (
                    (trunc_index + self.max_mel_tokens - 1)
                    if max_generate_length is None
                    else trunc_index + max_generate_length
                )
                self.gpt.store_mel_emb(embs, mask=mask)
                # while inputs.size < max_length and not (inputs == self.stop_mel_token).any():
                token_generator = generate_step(
                    prompt=tokens,
                    model=self.gpt,
                    logits_processors=logits_processors,
                    max_tokens=max_length - inputs.size,
                    sampler=sampler,
                )
                for token in token_generator:
                    next_token = mx.array([token])
                    tokens = mx.concatenate([tokens, next_token])
                    # print("next_token: ", next_token, "logprobs: ", logprobs)
                    if token == self.stop_mel_token:
                        break
                output.append(tokens[trunc_index:])
        mx.eval(output)
        if len(output) == 1:
            return torch.tensor(np.array(output[0], copy=False), device=text_inputs.device).unsqueeze(0)
        for i in range(len(output)):
            output[i] = torch.tensor(np.array(output[i], copy=False), device=text_inputs.device)
        return torch.nn.utils.rnn.pad_sequence(output, batch_first=True, padding_value=self.stop_mel_token)

def generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prefill_step_size: int = 2048,
) -> Generator[mx.array, None, None]:
    from mlx_lm.models.cache import KVCache
    y = prompt
    tokens = None
    num_layers = len(model.layers)
    prompt_cache = [KVCache() for _ in range(num_layers)]

    # sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    def _step(y):
        with mx.stream(generation_stream):
            logits = model(y[None], cache=prompt_cache)
            logits = logits[:, -1, :]

            if logits_processors:
                nonlocal tokens
                tokens = mx.concat([tokens, y]) if tokens is not None else y

                for processor in logits_processors:
                    logits = processor(tokens, logits)
            if sampler:
                y = sampler(logits)
            else:
                logprobs = logits - mx.logsumexp(logits, keepdims=True)
                y = mx.argmax(logprobs, axis=-1)
            return y

    with mx.stream(generation_stream):

        while y.size > prefill_step_size:
            model(y[:prefill_step_size][None], cache=prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            y = y[prefill_step_size:]
            mx.clear_cache()

        y = _step(y)

    mx.async_eval(y)
    n = 0
    while True:
        if n != max_tokens:
            next_y = _step(y)
            mx.async_eval(next_y)
        if n == 0:
            mx.eval(y)
        if n == max_tokens:
            break
        yield y.item()
        if n % 256 == 0:
            mx.clear_cache()
        y = next_y
        n += 1
