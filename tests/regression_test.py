# -*- coding: utf-8 -*-
from indextts.infer import IndexTTS
import os
import re

if __name__ == "__main__":
    """
    Test for 1.0 checkpoints:
    $ python tests/regression_test.py

    for 1.5 checkpoints:
    $ python tests/regression_test.py checkpoints-1.5
    """
    import sys
    sys.path.append("..")
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        model_dir = "checkpoints"
    prompt_wav="tests/sample_prompt.wav"
    tts = IndexTTS(cfg_path=os.path.join(model_dir, "config.yaml"), model_dir=model_dir, is_fp16=True, use_cuda_kernel=False)
    print("Model version: ", tts.model_version or "1.0")
    test_cases = [
        {"method": "infer", "text": "晕 XUAN4 是 一 种 GAN3 觉"},
        {"method": "infer", "text": "大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！"},
        {"method": "infer", "text": "There is a vehicle arriving in dock number 7?"},
        {"method": "infer", "text": "“我爱你！”的英语是“I love you!”"},
        {"method": "infer", "text": "Joseph Gordon-Levitt is an American actor"},
        {"method": "infer", "text": "约瑟夫·高登-莱维特是美国演员"},
        {"method": "infer", "text": "蒂莫西·唐纳德·库克（英文名：Timothy Donald Cook），通称蒂姆·库克（Tim Cook），现任苹果公司首席执行官。"},
        # 并行推理测试
        {"method": "infer_fast", "text": "亲爱的伙伴们，大家好！每一次的努力都是为了更好的未来，要善于从失败中汲取经验，让我们一起勇敢前行,迈向更加美好的明天！"},
        {"method": "infer_fast", "text": "The weather is really nice today, perfect for studying at home.Thank you!"},
        # 长文本推理测试
        {"method": "infer_fast", "text": "叶远随口答应一声，一定帮忙云云。教授看叶远的样子也知道，这事情多半是黄了。谁得到这样的东西也不会轻易贡献出来，这是很大的一笔财富。叶远回来后，又自己做了几次试验，发现空间湖水对一些外伤也有很大的帮助。找来一只断了腿的兔子，喝下空间湖水，一天时间，兔子就完全好了。还想多做几次试验，可是身边没有试验的对象，就先放到一边，了解空间湖水可以饮用，而且对人有利，这些就足够了。感谢您的收听，下期再见！"},
        {"method": "infer_fast", "text": "《盗梦空间》是由美国华纳兄弟影片公司出品的电影，由克里斯托弗·诺兰执导并编剧，莱昂纳多·迪卡普里奥、玛丽昂·歌迪亚、约瑟夫·高登-莱维特、艾利奥特·佩吉、汤姆·哈迪等联袂主演，2010年7月16日在美国上映，2010年9月1日在中国内地上映，2020年8月28日在中国内地重映。影片剧情游走于梦境与现实之间，被定义为“发生在意识结构内的当代动作科幻片”，讲述了由莱昂纳多·迪卡普里奥扮演的造梦师，带领特工团队进入他人梦境，从他人的潜意识中盗取机密，并重塑他人梦境的故事。"},
    ]

    os.makedirs("outputs", exist_ok=True)

    def safe_filename(text):
        return re.sub(r'[\\/:*?!"<>|\s]', '_', text)[:20] + ".wav"

    def get_generate_config(method):
        config = {
            "do_sample": True,
            "top_p": 0.8,
            "top_k": 30,
            "temperature": 1.0,
            "length_penalty": 0.0,
            "num_beams": 3,
            "repetition_penalty": 10.0,
            "max_text_tokens_per_sentence": 100,
        }
        if method == "infer_fast":
            config.update(
                {
                    "do_sample": True,
                    "num_beams": 1,
                    "sentences_bucket_max_size": 4,
                    "max_text_tokens_per_sentence": 80,
                }
            )
        return config
    for case in test_cases:
        method = getattr(tts, case["method"])
        output_path = os.path.join("outputs", safe_filename(case["text"]))
        method(audio_prompt=prompt_wav, text=case["text"], output_path=output_path, verbose=True, **get_generate_config(case["method"]))
