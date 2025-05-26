# -*- coding: utf-8 -*-
import asyncio
import json

import torchaudio
from indextts.infer import IndexTTS
import os
import re
import time
import torch

def safe_filename(text):
        return re.sub(r'[\\/:*?!"<>|\s]', "_", text)[:20] + ".wav"

def get_generate_config(tts, infer_mode):
    config = {
        "do_sample": True,
        "top_p": 0.8,
        "top_k": 30,
        "temperature": 1.0,
        "length_penalty": 0.0,
        "num_beams": 1,
        "repetition_penalty": 10.0,
        "max_text_tokens_per_sentence": 100,
        "max_mel_tokens": tts.gpt.max_mel_tokens,
    }
    if infer_mode == 1:
        config.update(
            {
                "num_beams": 1,
                "sentences_bucket_max_size": 4,
                "max_text_tokens_per_sentence": 120,
            }
        )
    return config

async def save_wav(wav, sr, output_path):
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, torch.tensor(wav, dtype=torch.int16), sr, channels_first=False)
    del wav

async def run_test_cases(tts: IndexTTS, test_cases, outputs_dir, verbose=False):
    prompt_wav = "tests/sample_prompt.wav"
    tasks = []
    results = []

    for case in test_cases:
        if case["infer_mode"] == 0:
            method = tts.infer
        elif case["infer_mode"] == 1:
            method = tts.infer_fast
        generate_config = get_generate_config(tts, case["infer_mode"])
        print(json.dumps(generate_config, indent=4))
        print("Synthesizing speech for text:", case["text"])
        start = time.perf_counter()
        sr, wav = method(
            audio_prompt=case.get("prompt_audio", prompt_wav),
            text=case["text"],
            verbose=verbose,
            **generate_config,
        )
        tts.torch_empty_cache()
        end = time.perf_counter()
        print(f"Take {end - start:.4f}s to synthesize speech.")
        wav_duration = wav.shape[0] / sr
        output_path = os.path.join(outputs_dir, safe_filename(case["text"]))
        tasks.append(save_wav(wav, sr, output_path))
        rtf = (end - start) / wav_duration
        results.append({
            "text": case["text"],
            "output": output_path,
            "wav duration": wav_duration,
            "RTF": f"{rtf:.4f}",
            "mode": case["infer_mode"],
        })

    # Await all save_wav tasks concurrently
    await asyncio.gather(*tasks)
    return results
        
async def main(args):
    import transformers
    model_dir = args.model_dir

    tts = IndexTTS(
        cfg_path=os.path.join(model_dir, "config.yaml"),
        model_dir=model_dir,
        is_fp16=not args.no_fp16,
        use_cuda_kernel=False,
        gpt_backend=args.gpt_backend,
    )
    print(
        "IndexTTS Model version: ",
        tts.model_version or "1.0",
        "transformers version: ",
        transformers.__version__,
        "torch version: ",
        torch.__version__,
        sep="\n",
    )
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    test_cases = [
        {"infer_mode": 0, "text": "There is a vehicle arriving in dock number 7?"},
        {"infer_mode": 0, "text": "“我爱你！”的英语是“I love you!”"},
        {"infer_mode": 0, "text": "Joseph Gordon-Levitt is an American actor"},
        {"infer_mode": 0, "text": "约瑟夫·高登-莱维特是美国演员"},
        {
            "infer_mode": 0,
            "text": "蒂莫西·唐纳德·库克（英文名：Timothy Donald Cook），通称蒂姆·库克（Tim Cook），现任苹果公司首席执行官。",
        },
        # 并行推理测试
        {
            "infer_mode": 1,
            "text": "亲爱的伙伴们，大家好！每一次的努力都是为了更好的未来，要善于从失败中汲取经验，让我们一起勇敢前行,迈向更加美好的明天！",
        },
        {"infer_mode": 1, "text": "The weather is really nice today, isn't it? Let's go for a walk in the park."},
        # 长文本推理测试
        {
            "infer_mode": 1,
            "text": "叶远随口答应一声，一定帮忙云云。教授看叶远的样子也知道，这事情多半是黄了。谁得到这样的东西也不会轻易贡献出来，这是很大的一笔财富。叶远回来后，又自己做了几次试验，发现空间湖水对一些外伤也有很大的帮助。找来一只断了腿的兔子，喝下空间湖水，一天时间，兔子就完全好了。还想多做几次试验，可是身边没有试验的对象，就先放到一边，了解空间湖水可以饮用，而且对人有利，这些就足够了。感谢您的收听，下期再见！",
        },
        {
            "infer_mode": 1,
            "text": "We have already extensively tested the current mobile graphics cards from Nvidia's RTX 5000 series, although availability is currently still very limited, especially for notebooks with the RTX 5070 Laptop and 5070 Ti Laptop. However, with the RTX 5060 Laptop, Nvidia is now launching the next mobile Blackwell GPU on the market, which is naturally positioned below the RTX 5070 Laptop in terms of performance and will therefore primarily be used in mainstream gaming laptops and fast multimedia laptops.",
        },
    ]
    
    # 额外的测试用例
    with open(os.path.join(tests_dir, "cases.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            prompt_audio = case.get("prompt_audio")
            if prompt_audio:
                prompt_audio = os.path.join(tests_dir, prompt_audio)
                case["prompt_audio"] = prompt_audio
            test_cases.append(case)

    outputs_dir = os.path.join("outputs", "tests", time.strftime("%H%M%S"))
    os.makedirs(outputs_dir, exist_ok=True)
    print("Output dir:", outputs_dir)
    results = await run_test_cases(tts, test_cases, outputs_dir, args.verbose)
    print("All tests done.")
    csv_file = os.path.join(outputs_dir, "test_results.csv")
    try:
        import pandas as pd

        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False)

        # 为 output 列加上 <audio> 标签
        def audio_tag(path):
            url = "file://" + os.path.abspath(path)
            return f'<audio controls src="{url}"></audio>'

        df["output"] = df["output"].apply(audio_tag)
        html_file = os.path.join(outputs_dir, "test_results.html")
        df.to_html(html_file, index=False, justify="center", escape=False)
        print(f"Results saved to {html_file}, {csv_file}")
    except ImportError:
        import csv

        async with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["infer mode", "output", "wav duration", "RTF"])
            for result in results:
                writer.writerow([result["mode"], result["RTF"], result["output"], result["wav duration"]])
        print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    """
    Test for 1.0 checkpoints:
    $ python tests/regression_test.py

    for 1.5 checkpoints:
    $ python tests/regression_test.py --model_dir checkpoints-1.5

    for mlx backend:
    $ python tests/regression_test.py --gpt_backend mlx
    """
    import sys
    import argparse

    sys.path.append("..")

    parser = argparse.ArgumentParser(description="IndexTTS regression test")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="checkpoints",
        help="Path to the model directory. Default is checkpoints.",
    )

    parser.add_argument(
        "--gpt_backend",
        type=str,
        default="transformers",
        choices=["transformers", "mlx"],
        help="Backend for GPT2 model.",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Enable verbose mode.",
    )
    parser.add_argument(
        "--no-fp16",
        default=False,
        action="store_true",
        help="Disable fp16 mode.",
    )
    args = parser.parse_args()
    
    asyncio.run(main(args))
