#!/usr/bin/env python3
"""
Test script to verify GPU memory leak fixes
Run this script and monitor GPU memory usage with nvidia-smi or similar tools
PYTHONPATH="$PYTHONPATH:." python3 tests/test_memory_fix.py.py IndexTTS-1.5
"""

import torch
import time
import gc

def get_gpu_memory_mb():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def test_multiple_inferences(model_dir, enable_emo_text=False):
    """Test multiple sequential inferences to detect memory leaks"""
    from indextts.infer_v2 import IndexTTS2

    print("Initializing IndexTTS2...")
    tts = IndexTTS2(
        cfg_path=f"{model_dir}/config.yaml",
        model_dir=model_dir,
        use_fp16=False,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_accel=False,
    )

    prompt_wavs = [
        "tests/sample_prompt.wav",
        "examples/voice_04.wav",
        "examples/voice_05.wav",
        "examples/voice_06.wav",
        "examples/voice_07.wav",
        "examples/voice_08.wav",
    ]
    test_texts = [
        "数到3就开始：1、2、3。嗨，大家好！今天我们要聊一个超酷话题——生成式AI！这是第一个测试文本。",
        "这是第二个测试文本，用于检测显存泄漏。",
        "第三次推理，观察显存变化。嗨，大家好！",
        "第四次推理，继续监控。模型会去寻找数据背后的“模式”",
        "第五次推理，看看显存是否稳定。生成式AI能帮我们写小说、画画，甚至设计新药和制作电影！",
        "第六次推理，希望没有显存泄漏问题。生成对抗网络(GAN)。训练过程中两个模型互相PK：一个负责造假图，一个负责鉴定真伪。",
        "第七次推理，继续观察显存使用情况。用一段较长的文本来测试，看看效果如何！那它是怎么做到的呢？秘密就在于“学习”和“模仿”。假设我要教一个模型画画。我会把地球上所有梵高、莫奈、毕加索等等画家的画都给它看。这些海量的数据，就是AI的“知识库”。",
        "第八次推理，继续监控显存。通过“深度学习”这个神奇工具，模型会去寻找数据背后的“模式”。找到了模式，模型就有了创造力。",
        "第九次推理，显存使用情况如何？比如我说：“帮我画一张像梵高画的、穿着宇航服的静香”，它就能把这些概念结合起来！",
        "第十次推理，最后一次测试显存稳定性。"
    ]
    emo_texts = [
        "严肃的",
        "唉声叹气",
        "打招呼的语气",
        "迟疑，思考",
        "兴奋的、炫耀",
        "疑惑不解"
    ]

    memory_usage = []

    print("\nStarting inference tests...")
    print("=" * 60)

    for i, text in enumerate(test_texts):
        emo_text = emo_texts[i % len(emo_texts)] if enable_emo_text else None
        audio_prompt = prompt_wavs[i % len(prompt_wavs)]
        print(f"\nTest {i+1}/5: {text}, Emo Text: {emo_text if emo_text else 'N/A'}, Audio Prompt: {audio_prompt}")
        print("[==] waiting 10 seconds before inference...")
        time.sleep(10)
        # Record memory before inference
        mem_before = get_gpu_memory_mb()

        # Run inference
        start_time = time.time()
        tts.infer(
            spk_audio_prompt=audio_prompt,
            text=text,
            output_path=f"test_output_{i + 1}.wav",
            use_emo_text=enable_emo_text,
            emo_text=emo_text,
            emo_alpha=0.8,
            verbose=False,
        )
        elapsed = time.time() - start_time

        # Record memory after inference
        mem_after = get_gpu_memory_mb()
        mem_diff = mem_after - mem_before

        memory_usage.append({
            'test': i+1,
            'before': mem_before,
            'after': mem_after,
            'diff': mem_diff,
            'time': elapsed
        })

        print(f"  Time: {elapsed:.2f}s")
        print(f"  GPU Memory: {mem_before:.1f} MB -> {mem_after:.1f} MB (diff: {mem_diff:+.1f} MB)")

    print("\n" + "=" * 60)
    print("\nMemory Usage Summary:")
    print("-" * 60)
    print(f"{'Test':<8} {'Before (MB)':<15} {'After (MB)':<15} {'Diff (MB)':<15}")
    print("-" * 60)

    for mem in memory_usage:
        print(f"{mem['test']:<8} {mem['before']:<15.1f} {mem['after']:<15.1f} {mem['diff']:<+15.1f}")

    # Calculate average memory increase (excluding first run which may include one-time allocations)
    if len(memory_usage) > 1:
        avg_increase = sum(m['diff'] for m in memory_usage[1:]) / (len(memory_usage) - 1)
        print("-" * 60)
        print(f"Average memory increase per inference (tests 2-5): {avg_increase:+.1f} MB")

        if avg_increase > 50:
            print("\n⚠️  WARNING: Significant memory increase detected!")
            print("   This may indicate a memory leak.")
        elif avg_increase > 10:
            print("\n⚠️  NOTICE: Small memory increase detected.")
            print("   This might be acceptable depending on the use case.")
        else:
            print("\n✓ Memory usage looks stable. No significant leak detected.")

    print("\nTest completed!")

if __name__ == "__main__":
    import sys
    if not torch.cuda.is_available():
        print("CUDA is not available. This test requires a GPU.")
        sys.exit(1)
    # print cuda info
    print("=" * 60)
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    print("Total GPU memory:", torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, "MB")
    print(f"Initial GPU memory usage: {get_gpu_memory_mb():.1f} MB")
    print("-" * 60)
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints"
    enable_emo_text = '--enable-emo-text' in sys.argv if len(sys.argv) > 1 else False
    test_multiple_inferences(model_dir, enable_emo_text)
