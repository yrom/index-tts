import os
import re
import time
from subprocess import CalledProcessError
from typing import Optional, Union

import sentencepiece as spm
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf

from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.gpt.model import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.feature_extractors import MelSpectrogramFeatures
from indextts.utils.common import tokenize_by_CJK_char

from indextts.utils.front import TextNormalizer

class IndexTTS:
    def __init__(
        self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True, device=None, use_cuda_kernel=None,
    ):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            is_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
        """
        if device is not None:
            self.device = device
            self.is_fp16 = False if device == "cpu" else is_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.is_fp16 = is_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif torch.mps.is_available():
            self.device = "mps"
            self.is_fp16 = is_fp16
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.is_fp16 = False
            self.use_cuda_kernel = False
            print(">> Be patient, it may take a while to run in CPU mode.")

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.is_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        
        # Comment-off to load the VQ-VAE model for debugging tokenizer
        #   https://github.com/index-tts/index-tts/issues/34
        #
        # from indextts.vqvae.xtts_dvae import DiscreteVAE
        # self.dvae = DiscreteVAE(**self.cfg.vqvae)
        # self.dvae_path = os.path.join(self.model_dir, self.cfg.dvae_checkpoint)
        # load_checkpoint(self.dvae, self.dvae_path)
        # self.dvae = self.dvae.to(self.device)
        # if self.is_fp16:
        #     self.dvae.eval().half()
        # else:
        #     self.dvae.eval()
        # print(">> vqvae weights restored from:", self.dvae_path)
        self.gpt = UnifiedVoice(**self.cfg.gpt)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.is_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        print(">> GPT weights restored from:", self.gpt_path)
        if self.is_fp16:
            try:
                import deepspeed
                use_deepspeed = True
            except (ImportError, OSError,CalledProcessError) as e:
                use_deepspeed = False
                print(f">> DeepSpeed加载失败，回退到标准推理: {e}")

            self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=True)
        else:
            self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=False)
        
        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.BigVGAN.alias_free_activation.cuda import load
                anti_alias_activation_cuda = load.load()
                print(">> Preload custom CUDA kernel for BigVGAN", anti_alias_activation_cuda)
            except:
                print(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.")
                self.use_cuda_kernel = False
        self.bigvgan = Generator(self.cfg.bigvgan, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan_path = os.path.join(self.model_dir, self.cfg.bigvgan_checkpoint)
        vocoder_dict = torch.load(self.bigvgan_path, map_location="cpu")
        self.bigvgan.load_state_dict(vocoder_dict["generator"])
        self.bigvgan = self.bigvgan.to(self.device)
        # remove weight norm on eval mode
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", self.bigvgan_path)
        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset['bpe_model'])
        self.tokenizer = spm.SentencePieceProcessor(model_file=self.bpe_path)
        print(">> bpe model loaded from:", self.bpe_path)
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        print(">> TextNormalizer loaded")

    def preprocess_text(self, text):
        # chinese_punctuation = "，。！？；：“”‘’（）【】《》"
        # english_punctuation = ",.!?;:\"\"''()[]<>"
        #
        # # 创建一个映射字典
        # punctuation_map = str.maketrans(chinese_punctuation, english_punctuation)

        # 使用translate方法替换标点符号
        # return text.translate(punctuation_map)
        return self.normalizer.infer(text)


    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        code_lens = []
        codes_list = []
        device = codes.device
        dtype = codes.dtype
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if self.cfg.gpt.stop_mel_token not in code:
                code_lens.append(len(code))
                len_ = len(code)
            else:
                # len_ = code.cpu().tolist().index(8193)+1
                len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0] + 1
                len_ = len_ - 2

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                code = code.cpu().tolist()
                ncode = []
                n = 0
                for k in range(0, len_):
                    if code[k] != silent_token:
                        ncode.append(code[k])
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode.append(code[k])
                        n += 1
                    # if (k == 0 and code[k] == 52) or (code[k] == 52 and code[k-1] == 52):
                    #    n += 1
                len_ = len(ncode)
                ncode = torch.LongTensor(ncode)
                codes_list.append(ncode.to(device, dtype=dtype))
                isfix = True
                #codes[i] = self.stop_mel_token
                #codes[i, 0:len_] = ncode
            else:
                codes_list.append(codes[i])
            code_lens.append(len_)

        codes = pad_sequence(codes_list, batch_first=True) if isfix else codes[:, :-2]
        code_lens = torch.LongTensor(code_lens).to(device, dtype=dtype)
        return codes, code_lens

    def split_sentences(self, text):
        """
        Split the text into sentences based on punctuation marks.
        """
        # 匹配标点符号（包括中英文标点）
        pattern = r'(?<=[.!?;。！？；])\s*'
        sentences = re.split(pattern, text)
        # 过滤掉空字符串和仅包含标点符号的字符串
        return [
            sentence.strip() for sentence in sentences if sentence.strip() and sentence.strip() not in {"'", ".", ","}
        ]

    def empty_cache(self):
        """
        Clear the GPU cache.
        """
        if self.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self.device.startswith('mps') and torch.mps.is_available():
            torch.mps.empty_cache()

    def load_audio(self, audio_path, verbose=False):
        audio, sr = torchaudio.load(audio_path)
        audio = torch.mean(audio, dim=0, keepdim=True).to(self.device)
        if audio.shape[0] > 1:
            audio = audio[0].unsqueeze(0)
        if sr != 24000:
            audio = torchaudio.transforms.Resample(sr, 24000).to(self.device)(audio)
            print(f">> Audio resample from {sr} to 24000")
        mel_spec = MelSpectrogramFeatures().to(self.device)
        cond_mel = mel_spec(audio).to(self.device)
        if verbose:
            print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)
        del audio
        return cond_mel

    def get_stats(self) -> Optional[dict]:
        return self.stats if hasattr(self, "stats") else None

    def infer(self, audio_prompt: Union[str, os.PathLike], text, output_path, verbose=False):
        print(">> start inference...")
        self.stats = {
            "gpt_gen_time": 0,
            "gpt_forward_time": 0,
            "bigvgan_time": 0,
        }
        if verbose:
            print(f"origin text:{text}")
        normalized_text = self.preprocess_text(text)
        print(f"normalized text:{normalized_text}")
        # load audio
        cond_mel = self.load_audio(audio_prompt, verbose=verbose)
        cond_mel_num_frames = cond_mel.shape[-1]
        if verbose:
            print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)
        start_time = time.perf_counter()
        wav_data, sampling_rate = self.infer_e2e(cond_mel, normalized_text, verbose=verbose)
        wav_length = wav_data.shape[-1] / sampling_rate
        end_time = time.perf_counter()
        print(
            f">> Reference audio length: {cond_mel_num_frames * 256 / sampling_rate:.2f} seconds",
            f"mel frames: {cond_mel_num_frames}",
        )
        stats = self.get_stats()
        for key, value in stats.items():
            print(f">> {key}: {value:.2f}")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")
        torchaudio.save(output_path, wav_data, sampling_rate)
        print(">> wav file saved to:", output_path)

        
    def infer_e2e(self, audio_mel: torch.Tensor, normalized_text, verbose=False) -> tuple[torch.Tensor, int]: 
        stats = self.get_stats()
        cond_mel_num_frames = audio_mel.shape[-1]
        assert cond_mel_num_frames > 0, "The input audio cannot be empty."
        auto_conditioning = audio_mel

        sentences = self.split_sentences(normalized_text)
        if verbose:
            print("sentences:", sentences)

        top_p = .8
        top_k = 30
        temperature = 1.0
        autoregressive_batch_size = 1
        length_penalty = 0.0
        num_beams = 3
        repetition_penalty = 10.0
        max_mel_tokens = 600
        sampling_rate = 24000
        # lang = "EN"
        # lang = "ZH"
        wavs = []
        gpt_gen_time = stats.get("gpt_gen_time", 0)
        gpt_forward_time = stats.get("gpt_forward_time", 0)
        bigvgan_time = stats.get("bigvgan_time", 0)

        for sent in sentences:
            self.empty_cache()
            # sent = " ".join([char for char in sent.upper()]) if lang == "ZH" else sent.upper()
            cleand_text = tokenize_by_CJK_char(sent)
            # cleand_text = "他 那 像 HONG3 小 孩 似 的 话 , 引 得 人 们 HONG1 堂 大 笑 , 大 家 听 了 一 HONG3 而 散 ."
            if verbose:
                print("cleand_text:", cleand_text)

            text_tokens = torch.tensor(self.tokenizer.EncodeAsIds(cleand_text),dtype=torch.int32, device=self.device).unsqueeze(0)
            # text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
            # text_tokens = F.pad(text_tokens, (1, 0), value=0)
            # text_tokens = F.pad(text_tokens, (0, 1), value=1)
            if verbose:
                print(text_tokens)
                print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                # debug tokenizer
                text_token_syms = self.tokenizer.IdToPiece(text_tokens[0].tolist())
                print(text_token_syms)

            # text_len = torch.IntTensor([text_tokens.size(1)], device=text_tokens.device)
            # print(text_len)
            from torch.profiler import profile, record_function
            m_start_time = time.perf_counter()
            with torch.no_grad():
                with record_function("gpt_inference"):
                    with torch.amp.autocast(self.device, enabled=self.dtype is not None, dtype=self.dtype):
                        codes = \
                            self.gpt.inference_speech(auto_conditioning, text_tokens,
                                                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]],
                                                                                      device=text_tokens.device),
                                                        # text_lengths=text_len,
                                                        do_sample=True,
                                                        top_p=top_p,
                                                        top_k=top_k,
                                                        temperature=temperature,
                                                        num_return_sequences=autoregressive_batch_size,
                                                        length_penalty=length_penalty,
                                                        num_beams=num_beams,
                                                        repetition_penalty=repetition_penalty,
                                                        max_generate_length=max_mel_tokens)
                gpt_gen_time += time.perf_counter() - m_start_time
                #codes = codes[:, :-2]
                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
                if verbose:
                    print(codes, type(codes))
                    print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                with record_function("remove_long_silence"):
                # remove ultra-long silence if exits
                # temporarily fix the long silence bug.
                    codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                m_start_time = time.perf_counter()
                
                with torch.amp.autocast(self.device, enabled=self.dtype is not None, dtype=self.dtype):
                    with record_function("gpt_forward"):
                        latent = \
                            self.gpt(auto_conditioning, text_tokens,
                                    torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                    code_lens*self.gpt.mel_length_compression,
                                    cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                    return_latent=True, clip_inputs=False)
                    gpt_forward_time += time.perf_counter() - m_start_time
                    m_start_time = time.perf_counter()
                    with record_function("bigvgan_infer"):
                        wav, _ = self.bigvgan(latent, auto_conditioning.transpose(1, 2))
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)
                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                if verbose:
                    print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
                # wavs.append(wav[:, :-512])
                wavs.append(wav)
                del text_tokens, codes, latent

        wav = torch.cat(wavs, dim=1)
        
        if stats:
            stats["gpt_gen_time"] = gpt_gen_time
            stats["gpt_forward_time"] = gpt_forward_time
            stats["bigvgan_time"] = bigvgan_time
        return (wav.type(torch.int16).cpu(), sampling_rate)


if __name__ == "__main__":
    prompt_wav = "testwav/input.wav"
    prompt_wav = "testwav/spk_1744181067_1.wav"
    text="晕 XUAN4 是 一 种 GAN3 觉"
    text = "There is a vehicle arriving in dock number 7?"
    text='大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！'

    tts = IndexTTS(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True, use_cuda_kernel=False)
    tts.infer(audio_prompt=prompt_wav, text=text, output_path="gen.wav", verbose=True)
