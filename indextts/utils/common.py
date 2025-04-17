import inspect
import os
import random
import re
import torch
import torch.nn as nn
from torch.profiler import record_function
from functools import wraps
import torchaudio

MATPLOTLIB_FLAG = False


def load_audio_wave(audiopath, sampling_rate=24000, mono=True, format=None, normalize=True):
    audio, sr = torchaudio.load(audiopath, format=format, normalize=normalize)
    if mono and audio.size(0) > 1:  # mix to mono
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != sampling_rate:
        resample = torchaudio.transforms.Resample(sr, sampling_rate, dtype=audio.dtype)
        audio = resample(audio)
    return audio


def tokenize_by_CJK_char(line: str) -> str:
    """
    Tokenize a line of text with CJK char.

    Note: All return charaters will be upper case.

    Example:
      input = "你好世界是 hello world 的中文"
      output = "你 好 世 界 是 HELLO WORLD 的 中 文"

    Args:
      line:
        The input text.

    Return:
      A new string tokenize by CJK char.
    """
    # The CJK ranges is from https://github.com/alvations/nltk/blob/79eed6ddea0d0a2c212c1060b477fc268fec4d4b/nltk/tokenize/util.py
    pattern = re.compile(
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
    )
    chars = pattern.split(line.strip().upper())
    return " ".join([w.strip() for w in chars if w.strip()])


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))


def detect_deepspeed():
    try:
        import platform

        if platform.system() != "Darwin":
            import deepspeed

            return torch.cuda.is_available()
    except Exception as e:
        print(f">> DeepSpeed加载失败，回退到标准推理: {e}")

    return False


import wrapt
import inspect

import inspect


class CallableWrapper(wrapt.ObjectProxy):
    def __init__(self, wrapped, wrapper):
        super(CallableWrapper, self).__init__(wrapped)
        self._self_wrapper = wrapper

    def __call__(self, *args, **kwargs):
        return self._self_wrapper(self.__wrapped__, args, kwargs)


class BoundCallableWrapper(wrapt.ObjectProxy):
    def __init__(self, wrapped, wrapper):
        super(BoundCallableWrapper, self).__init__(wrapped)
        self._self_wrapper = wrapper

    def __get__(self, instance, owner):
        return self

    def __call__(self, *args, **kwargs):
        
        return self._self_wrapper(self.__wrapped__, args, kwargs)

class ModuleWrapper(wrapt.ObjectProxy):

    def __init__(self, wrapped: nn.Module):
        super(ModuleWrapper, self).__init__(wrapped)
        #print(f"Wrapping module: {wrapped.__class__.__name__}")
        def __method_wrapper(wrapped, args, kwargs):
            #print(f"Calling {wrapped.__name__}")
            with record_function(f"{self.__wrapped__.__class__.__name__}.{wrapped.__name__}"):
                return wrapped(*args, **kwargs)

        child_modules = inspect.getmembers(wrapped, predicate=lambda m: isinstance(m, nn.Module))
        for n, child in child_modules:
            if (
                child != wrapped
                and isinstance(child, nn.Module)
                # and not isinstance(child, ModuleWrapper)
                and not isinstance(child, wrapt.ObjectProxy)
                # and not isinstance(child, nn.ModuleList)
                # and not child.__class__.__name__.startswith("ConvNd")
            ):
                # Wrap the child module with ModuleWrapper
                wrapped_child = ModuleWrapper(child)
                print (f"Wrapping child module: {wrapped_child.__class__.__name__}")
                setattr(wrapped, n, wrapped_child)

        wrapped_methods = inspect.getmembers(wrapped, predicate=inspect.ismethod)
        for name, method in wrapped_methods:
            if name.startswith("_") or name.startswith("named_") or name in ["forward", "to", "train", "eval", "type", "cuda", "cpu", "float", "double", "half"]:
                continue
            if isinstance(method, CallableWrapper):
                # print(f"Skipping already wrapped method: {name} in {wrapped.__class__.__name__}")
                continue
            # Wrap the method with a CallableWrapper
            wrapped_method = CallableWrapper(method, __method_wrapper)
            setattr(wrapped, name, wrapped_method)
    def __call__(self, *args, **kwargs):
        #print(f"Calling {self.__wrapped__.__class__.__name__} with args: {args}, kwargs: {kwargs}")
        with record_function(self.__wrapped__.__class__.__name__):
            return self.__wrapped__(*args, **kwargs)

def profile_module(module: nn.Module) -> nn.Module:
    return ModuleWrapper(module)
