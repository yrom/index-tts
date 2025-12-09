# -*- coding: utf-8 -*-
from functools import lru_cache
import os
import traceback
import re
from typing import List, Union, overload
import warnings
from indextts.utils.common import tokenize_by_CJK_char, de_tokenized_by_CJK_char
from sentencepiece import SentencePieceProcessor


class TextNormalizer:
    def __init__(self, enable_glossary=False):
        self.zh_normalizer = None
        self.en_normalizer = None
        self.char_rep_map = {
            "：": ",",
            "；": ",",
            ";": ",",
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "\n": " ",
            "·": "-",
            "、": ",",
            "...": "…",
            ",,,": "…",
            "，，，": "…",
            "……": "…",
            "“": "'",
            "”": "'",
            '"': "'",
            "‘": "'",
            "’": "'",
            "（": "'",
            "）": "'",
            "(": "'",
            ")": "'",
            "《": "'",
            "》": "'",
            "【": "'",
            "】": "'",
            "[": "'",
            "]": "'",
            "—": "-",
            "～": "-",
            "~": "-",
            "「": "'",
            "」": "'",
            ":": ",",
        }
        self.zh_char_rep_map = {
            "$": ".",
            **self.char_rep_map,
        }
        self.enable_glossary = enable_glossary
        # 术语词汇表：用户可自定义专业术语的读法
        # 格式: {"原始术语": {"en": "英文读法", "zh": "中文读法"}}
        # "M.2": {"en": "M dot two", "zh": "M 二"},
        # "PCIe 5.0": {"en": "PCIE five", "zh": "PCIE 五点零"},
        # "PCIe 4.0": {"en": "PCIE four", "zh": "PCIE 四点零"},
        # "AHCI": "A H C I",
        # "TTS": "T T S",
        # "Inc.": {"en": "Ink"},
        # ".json": {"en": " dot Jay-Son", "zh": "点 Jay-Son"},
        # "C++": {"en": "C plus plus", "zh": "C 加加"},
        # "C#": "C sharp"
        # self.term_glossary = {
        #     "C++": {"en": "C plus plus", "zh": "C 加加"},
        #     "C#": "C sharp",
        #     "CMake": "C Make",
        # }
        self.term_glossary = dict()

    def match_email(self, email):
        # 正则表达式匹配邮箱格式：数字英文@数字英文.英文
        pattern = r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$"
        return re.match(pattern, email) is not None

    PINYIN_TONE_PATTERN = r"(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])"
    """
    匹配拼音声调格式：pinyin+数字，声调1-5，5表示轻声
    例如：xuan4, jve2, ying1, zhong4, shang5
    不匹配：beta1, voice2
    """
    NAME_PATTERN = r"[\u4e00-\u9fff]+(?:[-·—][\u4e00-\u9fff]+){1,2}"
    """
    匹配人名，格式：中文·中文，中文·中文-中文
    例如：克里斯托弗·诺兰，约瑟夫·高登-莱维特
    """

    TECH_TERM_PATTERN = r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+"
    """
    匹配技术术语，格式：字母开头+(字母或数字)*+(-字母或数字)+
    例如：GPT-5-nano, F5-TTS, Fish-Speech, GPT-5, CosyVoice-2
    必须以字母开头，避免匹配纯数字（如电话号码 135-4567-8900）
    用于保护连字符结构，防止中文normalizer将连字符解析为减号（如"负五减"）
    """

    # 匹配常见英语缩写 's，仅用于替换为 is，不匹配所有 's
    ENGLISH_CONTRACTION_PATTERN = r"(what|where|who|which|how|t?here|it|s?he|that|this)'s"


    def use_chinese(self, s):
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", s))
        has_alpha = bool(re.search(r"[a-zA-Z]", s))
        is_email = self.match_email(s)
        if has_chinese or not has_alpha or is_email:
            return True

        has_pinyin = bool(re.search(TextNormalizer.PINYIN_TONE_PATTERN, s, re.IGNORECASE))
        return has_pinyin

    def load(self):
        # print(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        # sys.path.append(model_dir)
        import platform
        if self.zh_normalizer is not None and self.en_normalizer is not None:
            return
        if platform.system() != "Linux":  # Mac and Windows
            from wetext import Normalizer

            self.zh_normalizer = Normalizer(remove_erhua=False, lang="zh", operator="tn")
            self.en_normalizer = Normalizer(lang="en", operator="tn")
        else:
            from tn.chinese.normalizer import Normalizer as NormalizerZh
            from tn.english.normalizer import Normalizer as NormalizerEn
            # use new cache dir for build tagger rules with disable remove_interjections and remove_erhua
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tagger_cache")
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                with open(os.path.join(cache_dir, ".gitignore"), "w") as f:
                    f.write("*\n")
            self.zh_normalizer = NormalizerZh(
                cache_dir=cache_dir, remove_interjections=False, remove_erhua=False, overwrite_cache=False
            )
            self.en_normalizer = NormalizerEn(overwrite_cache=False)

    def normalize(self, text: str) -> str:
        if not self.zh_normalizer or not self.en_normalizer:
            print("Error, text normalizer is not initialized !!!")
            return ""
        if self.use_chinese(text):
            text = re.sub(TextNormalizer.ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
            # 应用术语词汇表（优先级最高，在所有保护之前）
            if self.enable_glossary:
                text = self.apply_glossary_terms(text, lang="zh")
            # 保护技术术语（如 GPT-5-nano）避免被中文normalizer错误处理
            replaced_text, tech_list = self.save_tech_terms(text.rstrip())
            replaced_text, pinyin_list = self.save_pinyin_tones(replaced_text)

            replaced_text, original_name_list = self.save_names(replaced_text)
            try:
                result = self.zh_normalizer.normalize(replaced_text)
            except Exception:
                result = ""
                print(traceback.format_exc())
            # 恢复人名
            result = self.restore_names(result, original_name_list)
            # 恢复拼音声调
            result = self.restore_pinyin_tones(result, pinyin_list)
            # 恢复技术术语
            result = self.restore_tech_terms(result, tech_list)
            pattern = re.compile("|".join(re.escape(p) for p in self.zh_char_rep_map.keys()))
            result = pattern.sub(lambda x: self.zh_char_rep_map[x.group()], result)
        else:
            try:
                text = re.sub(TextNormalizer.ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
                # 应用术语词汇表（优先级最高，在所有保护之前）
                if self.enable_glossary:
                    text = self.apply_glossary_terms(text, lang="en")
                # 保护技术术语（如 GPT-5-Nano）避免被英文normalizer错误处理
                replaced_text, tech_list = self.save_tech_terms(text)
                result = self.en_normalizer.normalize(replaced_text)
                # 恢复技术术语
                result = self.restore_tech_terms(result, tech_list)
            except Exception:
                result = text
                print(traceback.format_exc())
            pattern = re.compile("|".join(re.escape(p) for p in self.char_rep_map.keys()))
            result = pattern.sub(lambda x: self.char_rep_map[x.group()], result)
        return result

    def correct_pinyin(self, pinyin: str):
        """
        将 jqx 的韵母为 u/ü 的拼音转换为 v
        如：ju -> jv , que -> qve, xün -> xvn
        """
        if pinyin[0] not in "jqxJQX":
            return pinyin
        # 匹配 jqx 的韵母为 u/ü 的拼音
        pattern = r"([jqx])[uü](n|e|an)*(\d)"
        repl = r"\g<1>v\g<2>\g<3>"
        pinyin = re.sub(pattern, repl, pinyin, flags=re.IGNORECASE)
        return pinyin.upper()

    def save_names(self, original_text):
        """
        替换人名为占位符 <n_a>、 <n_b>, ...
        例如：克里斯托弗·诺兰 -> <n_a>
        """
        # 人名
        name_pattern = re.compile(TextNormalizer.NAME_PATTERN, re.IGNORECASE)
        original_name_list = re.findall(name_pattern, original_text)
        if len(original_name_list) == 0:
            return (original_text, None)
        original_name_list = list(set("".join(n) for n in original_name_list))
        transformed_text = original_text
        # 替换占位符 <n_a>、 <n_b>, ...
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(name, f"<n_{number}>")

        return transformed_text, original_name_list

    def restore_names(self, normalized_text, original_name_list):
        """
        恢复人名为原来的文字
        例如：<n_a> -> original_name_list[0]
        """
        if not original_name_list or len(original_name_list) == 0:
            return normalized_text

        transformed_text = normalized_text
        # 替换为占位符 <n_a>、 <n_b>, ...
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(f"<n_{number}>", name)
        return transformed_text

    def save_tech_terms(self, original_text):
        """
        保护技术术语中的连字符，防止被中文normalizer解析为减号
        策略：将术语中的连字符替换为特殊占位符<H>，数字仍可被正常处理
        例如：GPT-5-nano -> GPT<H>5<H>nano，然后 5 被转换为 五
        最终恢复为：GPT-五-nano
        """
        tech_pattern = re.compile(TextNormalizer.TECH_TERM_PATTERN)
        original_tech_list = tech_pattern.findall(original_text)
        if len(original_tech_list) == 0:
            return (original_text, None)

        # 去重并按长度降序排列（避免短匹配先替换导致问题）
        original_tech_list = sorted(set(original_tech_list), key=len, reverse=True)
        transformed_text = original_text

        # 将术语中的连字符替换为占位符 <H>
        for term in original_tech_list:
            # 将 GPT-5-nano 替换为 GPT<H>5<H>nano
            protected_term = term.replace("-", "<H>")
            transformed_text = transformed_text.replace(term, protected_term)

        return transformed_text, original_tech_list

    def restore_tech_terms(self, normalized_text, original_tech_list):
        """
        恢复技术术语中的连字符
        将占位符 <H> 恢复为连字符 -
        同时清理 normalizer 可能在占位符周围添加的多余空格
        """
        if not original_tech_list or len(original_tech_list) == 0:
            return normalized_text

        # 清理 <H> 周围可能的空格，然后恢复为连字符
        # 处理模式: " <H> " -> "-", " <H>" -> "-", "<H> " -> "-", "<H>" -> "-"
        transformed_text = re.sub(r'\s*<H>\s*', '-', normalized_text)
        return transformed_text

    def apply_glossary_terms(self, text, lang="zh"):
        """
        应用术语词汇表，将专业术语替换为对应语言的读法

        Args:
            text: 待处理文本
            lang: 语言类型 "zh" 或 "en"

        Returns:
            处理后的文本

        Example:
            "M.2 NVMe SSD" -> (zh) "M 二 NVMe SSD"
            "M.2 NVMe SSD" -> (en) "M dot two NVMe SSD"
        """
        if not self.term_glossary:
            return text

        # 按术语长度降序排列，避免短术语先匹配导致长术语无法匹配
        # 例如："PCIe 5.0" 应该在 "PCIe" 之前匹配
        sorted_terms = sorted(self.term_glossary.keys(), key=len, reverse=True)
        @lru_cache(maxsize=42)
        def get_term_pattern(term: str):
            return re.compile(re.escape(term), re.IGNORECASE)
        transformed_text = text
        for term in sorted_terms:
            term_value = self.term_glossary[term]
            if isinstance(term_value, dict):
                replacement = term_value.get(lang, term_value.get(lang, term))
            else:
                replacement = term_value
            # 使用正则进行大小写不敏感的替换
            pattern = get_term_pattern(term)
            transformed_text = pattern.sub(replacement, transformed_text)

        return transformed_text

    def load_glossary(self, glossary_dict):
        """
        加载外部术语词汇表

        Args:
            glossary_dict: 术语词典，格式为 {"术语": {"en": "英文读法", "zh": "中文读法"}}

        Example:
            normalizer.load_glossary({
                "M.2": {"en": "M dot two", "zh": "M 二"},
                "PCIe": {"en": "PCIE", "zh": "PCIE"}
            })
        """
        if glossary_dict and isinstance(glossary_dict, dict):
            self.term_glossary.update(glossary_dict)

    def load_glossary_from_yaml(self, glossary_path):
        """
        从 YAML 文件加载术语词汇表

        Args:
            glossary_path: YAML 文件路径

        Example:
            normalizer.load_glossary_from_yaml("checkpoints/glossary.yaml")

        YAML 文件格式:
            M.2:
              en: M dot two
              zh: M 二
            NVMe: N-V-M-E  # 中英文相同读法
        """
        if glossary_path and os.path.exists(glossary_path):
            import yaml
            with open(glossary_path, 'r', encoding='utf-8') as f:
                external_glossary = yaml.safe_load(f)
                if external_glossary and isinstance(external_glossary, dict):
                    self.term_glossary = external_glossary
                    return True
        return False

    def save_glossary_to_yaml(self, glossary_path):
        """
        保存术语词汇表到 YAML 文件

        Args:
            glossary_path: YAML 文件路径
        """
        import yaml
        with open(glossary_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.term_glossary, f, allow_unicode=True, default_flow_style=False)

    def save_pinyin_tones(self, original_text):
        """
        替换拼音声调为占位符 <pinyin_a>, <pinyin_b>, ...
        例如：xuan4 -> <pinyin_a>
        """
        # 声母韵母+声调数字
        origin_pinyin_pattern = re.compile(TextNormalizer.PINYIN_TONE_PATTERN, re.IGNORECASE)
        original_pinyin_list = re.findall(origin_pinyin_pattern, original_text)
        if len(original_pinyin_list) == 0:
            return (original_text, None)
        original_pinyin_list = list(set("".join(p) for p in original_pinyin_list))
        transformed_text = original_text
        # 替换为占位符 <pinyin_a>, <pinyin_b>, ...
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(pinyin, f"<pinyin_{number}>")

        # print("original_text: ", original_text)
        # print("transformed_text: ", transformed_text)
        return transformed_text, original_pinyin_list

    def restore_pinyin_tones(self, normalized_text, original_pinyin_list):
        """
        恢复拼音中的音调数字（1-5）为原来的拼音
        例如：<pinyin_a> -> original_pinyin_list[0]
        """
        if not original_pinyin_list or len(original_pinyin_list) == 0:
            return normalized_text

        transformed_text = normalized_text
        # 替换占位符 <pinyin_a>, <pinyin_b>, ...
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            pinyin = self.correct_pinyin(pinyin)
            transformed_text = transformed_text.replace(f"<pinyin_{number}>", pinyin)
        # print("normalized_text: ", normalized_text)
        # print("transformed_text: ", transformed_text)
        return transformed_text


class TextTokenizer:
    def __init__(self, vocab_file: str, normalizer: TextNormalizer = None):
        self.vocab_file = vocab_file
        self.normalizer = normalizer

        if self.vocab_file is None:
            raise ValueError("vocab_file is None")
        if not os.path.exists(self.vocab_file):
            raise ValueError(f"vocab_file {self.vocab_file} does not exist")
        if self.normalizer:
            self.normalizer.load()
        # 加载词表
        self.sp_model = SentencePieceProcessor(model_file=self.vocab_file)

        self.pre_tokenizers = [
            # 预处理器
            tokenize_by_CJK_char,
        ]

    @property
    def vocab_size(self):
        return self.sp_model.GetPieceSize()

    @property
    def unk_token(self):
        return "<unk>"

    @property
    def pad_token(self):
        return None

    @property
    def bos_token(self):
        return "<s>"

    @property
    def eos_token(self):
        return "</s>"

    @property
    def pad_token_id(self):
        return -1

    @property
    def bos_token_id(self):
        return 0

    @property
    def eos_token_id(self):
        return 1

    @property
    def unk_token_id(self):
        return self.sp_model.unk_id()

    @property
    def special_tokens_map(self):
        return {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab

    @overload
    def convert_ids_to_tokens(self, ids: int) -> str: ...

    @overload
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]: ...

    def convert_ids_to_tokens(self, ids: Union[List[int], int]):
        return self.sp_model.IdToPiece(ids)

    def convert_tokens_to_ids(self, tokens: Union[List[str], str]) -> List[int]:
        if isinstance(tokens, str):
            tokens = [tokens]
        return [self.sp_model.PieceToId(token) for token in tokens]

    def tokenize(self, text: str) -> List[str]:
        return self.encode(text, out_type=str)

    def encode(self, text: str, **kwargs):
        if len(text) == 0:
            return []
        if len(text.strip()) == 1:
            return self.sp_model.Encode(text, out_type=kwargs.pop("out_type", int), **kwargs)
        # 预处理
        if self.normalizer:
            text = self.normalizer.normalize(text)
        if len(self.pre_tokenizers) > 0:
            for pre_tokenizer in self.pre_tokenizers:
                text = pre_tokenizer(text)
        return self.sp_model.Encode(text, out_type=kwargs.pop("out_type", int), **kwargs)

    def batch_encode(self, texts: List[str], **kwargs):
        # 预处理
        if self.normalizer:
            texts = [self.normalizer.normalize(text) for text in texts]
        if len(self.pre_tokenizers) > 0:
            for pre_tokenizer in self.pre_tokenizers:
                texts = [pre_tokenizer(text) for text in texts]
        return self.sp_model.Encode(texts, out_type=kwargs.pop("out_type", int), **kwargs)

    def decode(self, ids: Union[List[int], int], do_lower_case=False, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        decoded = self.sp_model.Decode(ids, out_type=kwargs.pop("out_type", str), **kwargs)
        return de_tokenized_by_CJK_char(decoded, do_lower_case=do_lower_case)

    @staticmethod
    def _is_quote_token(token: str) -> bool:
        """检查是否为引号token"""
        return token in ["'", "▁'", '"', '▁"']

    @staticmethod
    def _count_quotes_in_segment(segment: List[str]) -> int:
        """统计segment中引号的数量"""
        return sum(1 for token in segment if TextTokenizer._is_quote_token(token))

    @staticmethod
    def _is_double_hyphen_at(tokens: List[str], pos: int) -> bool:
        """
        检查当前位置是否为双连字符模式
        双连字符模式: "▁" + "-" + "-" (三个token) 或在某个"-"位置且前一个也是"-"

        Args:
            tokens: token列表
            pos: 当前位置

        Returns:
            True表示当前位置是双连字符模式的一部分
        """
        if pos < 0 or pos >= len(tokens):
            return False

        # 检查模式1: ▁ + - + - (在第二个"-"位置检测)
        if tokens[pos] == "-" and pos >= 2:
            # 检查前面是否为: ▁, -
            if tokens[pos - 1] == "-" and tokens[pos - 2] == "▁":
                return True

        # 检查模式2: ▁ + - + - (在第一个"-"位置检测，需要向后看)
        if tokens[pos] == "-" and pos >= 1 and pos + 1 < len(tokens):
            # 检查是否为: ▁, -, -
            if tokens[pos - 1] == "▁" and tokens[pos + 1] == "-":
                return True

        return False

    @staticmethod
    def _is_name_hyphen(tokens: List[str], pos: int) -> bool:
        """
        判断当前位置的连字符是否为人名连接符
        人名连接符模式: ▁ + - + ▁ (三个token，单连字符)
        句子分隔符模式: ▁ + - + - (三个token，双连字符)

        Args:
            tokens: token列表
            pos: 当前'-'的位置

        Returns:
            True表示是人名连接符(不应该切分), False表示可以切分
        """
        if pos < 0 or pos >= len(tokens):
            return False

        current_token = tokens[pos]

        # 如果不是连字符token，返回False
        if current_token != "-":
            return False

        # 检查是否为双连字符模式 (▁ + - + -)
        if TextTokenizer._is_double_hyphen_at(tokens, pos):
            # 这是双连字符，可以切分
            return False

        # 检查前后是否有token
        has_prev = pos > 0
        has_next = pos < len(tokens) - 1

        if not (has_prev and has_next):
            # 如果在开头或结尾，认为不是人名连接符
            return False

        prev_token = tokens[pos - 1]
        next_token = tokens[pos + 1]

        # 人名连接符模式: ▁ + - + ▁
        # 前后都是空格token，说明这是人名连接符
        if prev_token == "▁" and next_token == "▁":
            return True

        # 其他情况认为可以切分
        return False

    @staticmethod
    def split_segments_by_token(
        tokenized_str: List[str],
        split_tokens: List[str],
        max_text_tokens_per_segment: int,
        quick_streaming_tokens: int = 0
    ) -> List[List[str]]:
        """
        将tokenize后的结果按语义优先级进行分割

        分割优先级:
        1. 强语义边界: 句号、问号、感叹号、省略号 (split_tokens传入的)
        2. 中等语义边界: 分号、双连字符(--)
        3. 弱语义边界: 逗号 (但需要考虑引号配对)
        4. 如果超过长度限制，强制按长度切分

        特殊处理:
        - 引号需要成对出现在同一segment中
        - 单个连字符(▁-)可能是人名连接符，不切分
        - 双连字符(▁--)是明确的分隔符，可以切分

        Args:
            tokenized_str: tokenize后的字符串列表
            split_tokens: 主要分割token (通常是强语义边界如., !, ?等)
            max_text_tokens_per_segment: 每个segment的最大token数
            quick_streaming_tokens: 快速流式输出的token数阈值

        Returns:
            分割后的segment列表
        """
        if len(tokenized_str) == 0:
            return []

        # 定义分割优先级
        strong_boundaries = set(split_tokens)  # 强边界: ., !, ?, ...等
        medium_boundaries = {";", "▁;"}  # 中等边界: 分号 (双连字符单独处理)
        weak_boundaries = {",", "▁,"}  # 弱边界: 逗号

        def should_split_at(pos: int, tokens: List[str], boundary_set: set) -> bool:
            """
            判断是否应该在某个位置切分

            特殊处理:
            - 双连字符(▁ + - + -): 在第二个"-"位置切分(作为中等边界)
            - 单连字符(▁ + - + ▁): 如果是人名连接符则不切分
            """
            if pos < 0 or pos >= len(tokens):
                return False

            token = tokens[pos]

            # 特殊处理: 检查是否为双连字符的第二部分 (▁ + - + -)
            # 在第二个"-"处切分，作为中等边界处理
            if token == "-" and TextTokenizer._is_double_hyphen_at(tokens, pos):
                # 这是双连字符，按中等边界处理
                return boundary_set == medium_boundaries

            # 检查是否在边界集合中
            if token not in boundary_set:
                return False

            # 特殊处理: 如果是连字符，检查是否为人名连接符
            if token == "-":
                # 单连字符需要判断是否为人名连接符
                return not TextTokenizer._is_name_hyphen(tokens, pos)

            # 其他边界token正常切分
            return True

        def find_best_split_point(segment: List[str], max_len: int) -> int:
            """
            在segment中找到最佳切分点
            优先级: 强边界 > 中等边界 > 弱边界(考虑引号) > 强制切分

            Returns:
                最佳切分位置(切分后，segment[:pos+1]为第一段)，-1表示不切分
            """
            if len(segment) <= max_len:
                return -1

            # 尝试在max_len范围内找到最佳切分点
            # 从后往前找，优先使用靠近max_len的切分点

            # 1. 尝试强边界
            for i in range(min(len(segment), max_len) - 1, 1, -1):
                if should_split_at(i, segment, strong_boundaries) and i > 2:
                    return i

            # 2. 尝试中等边界
            for i in range(min(len(segment), max_len) - 1, 1, -1):
                if should_split_at(i, segment, medium_boundaries) and i > 2:
                    return i

            # 3. 尝试弱边界(逗号)，但要考虑引号配对
            # 首先找到所有可能的切分点
            candidate_split_points = []
            for i in range(min(len(segment), max_len) - 1, 1, -1):
                if should_split_at(i, segment, weak_boundaries) and i > 2:
                    # 检查切分后引号是否配对
                    left_part = segment[:i+1]
                    quote_count = TextTokenizer._count_quotes_in_segment(left_part)
                    if quote_count % 2 == 0:  # 引号成对
                        candidate_split_points.append(i)

            # 找到所有人名连接符的位置
            name_hyphen_positions = []
            for j in range(len(segment)):
                if segment[j] == "-" and TextTokenizer._is_name_hyphen(segment, j):
                    name_hyphen_positions.append(j)

            # 选择最佳切分点：优先选择不会切断人名的点
            for split_pos in candidate_split_points:
                # 检查这个切分点是否会把人名分开
                # 如果人名连接符在切分点前后15个token内，则可能切断人名
                will_split_name = False
                for name_pos in name_hyphen_positions:
                    distance = abs(split_pos - name_pos)
                    # 如果距离小于15，说明可能会切断人名
                    if distance < 15:
                        will_split_name = True
                        break

                if not will_split_name:
                    return split_pos

            # 如果所有候选点都会切断人名，返回最后一个(最靠近max_len的)
            if candidate_split_points:
                return candidate_split_points[0]

            # 4. 强制切分: 如果都找不到，在max_len处强制切
            if len(segment) > max_len:
                warnings.warn(
                    f"Force splitting segment at max_len={max_len}. "
                    f"Segment length: {len(segment)}, tokens: {segment[:20]}...",
                    RuntimeWarning,
                )
                return max_len - 1

            return -1

        # 第一阶段: 按强边界进行初步分割
        initial_segments = []
        current_segment = []

        i = 0
        while i < len(tokenized_str):
            token = tokenized_str[i]
            current_segment.append(token)

            # 检查是否在强边界处切分
            if should_split_at(i, tokenized_str, strong_boundaries) and len(current_segment) > 2:
                # 检查下一个token是否是引号(避免在"句子."后的"处切分)
                if i + 1 < len(tokenized_str) and TextTokenizer._is_quote_token(tokenized_str[i + 1]):
                    # 将引号也包含进来
                    current_segment.append(tokenized_str[i + 1])
                    i += 1

                initial_segments.append(current_segment)
                current_segment = []

            i += 1

        # 添加最后一个segment
        if current_segment:
            initial_segments.append(current_segment)

        # 第二阶段: 处理超长segment，进行二次切分
        final_segments = []
        for segment in initial_segments:
            if len(segment) == 0:
                continue

            # 如果segment超长，递归切分
            while len(segment) > max_text_tokens_per_segment:
                split_pos = find_best_split_point(segment, max_text_tokens_per_segment)

                if split_pos <= 0:
                    # 无法找到合适切分点，强制切分
                    final_segments.append(segment[:max_text_tokens_per_segment])
                    segment = segment[max_text_tokens_per_segment:]
                else:
                    # 在找到的切分点处切分
                    final_segments.append(segment[:split_pos + 1])
                    segment = segment[split_pos + 1:]

            # 添加剩余部分
            if segment:
                final_segments.append(segment)

        # 第三阶段: 智能合并短segment
        merged_segments = []
        total_tokens = 0

        for segment in final_segments:
            total_tokens += len(segment)

            if len(segment) == 0:
                continue

            if len(merged_segments) == 0:
                merged_segments.append(segment)
            else:
                prev_segment = merged_segments[-1]
                combined_len = len(prev_segment) + len(segment)

                # 合并条件:
                # 1. 总长度不超过max_len
                # 2. 满足以下任一条件:
                #    a) 已经积累了足够的tokens(quick_streaming) 且合并后不超限
                #    b) 当前segment很短(< max_len/2)
                should_merge = False

                if combined_len <= max_text_tokens_per_segment:
                    if total_tokens > quick_streaming_tokens:
                        # 已积累足够tokens，可以合并
                        should_merge = True
                    elif len(segment) <= max_text_tokens_per_segment / 2:
                        # 当前segment很短，合并以提高效率
                        should_merge = True

                if should_merge:
                    merged_segments[-1] = prev_segment + segment
                else:
                    merged_segments.append(segment)

        return merged_segments

    punctuation_marks_tokens = [
        ".",
        "!",
        "?",
        "▁.",
        # "▁!", # unk
        "▁?",
        "▁...", # ellipsis
    ]
    def split_segments(self, tokenized: List[str], max_text_tokens_per_segment=120, quick_streaming_tokens = 0) -> List[List[str]]:
        return TextTokenizer.split_segments_by_token(
            tokenized, self.punctuation_marks_tokens, max_text_tokens_per_segment=max_text_tokens_per_segment, quick_streaming_tokens = quick_streaming_tokens
        )


if __name__ == "__main__":
    # 测试程序

    text_normalizer = TextNormalizer(enable_glossary=True)

    cases = [
        "IndexTTS 正式发布1.0版本了，效果666",
        "晕XUAN4是一种GAN3觉",
        "我爱你！",
        "I love you!",
        "“我爱你”的英语是“I love you”",
        "2.5平方电线",
        "共465篇，约315万字",
        "2002年的第一场雪，下在了2003年",
        "速度是10km/h",
        "现在是北京时间2025年01月11日 20:00",
        "他这条裤子是2012年买的，花了200块钱",
        "电话：135-4567-8900",
        "1键3连",
        "他这条视频点赞3000+，评论1000+，收藏500+",
        "这是1024元的手机，你要吗？",
        "受不liao3你了",
        "“衣裳”不读衣chang2，而是读衣shang5",
        "最zhong4要的是：不要chong2蹈覆辙",
        "不zuo1死就不会死",
        "See you at 8:00 AM",
        "8:00 AM 开会",
        "Couting down 3, 2, 1, go!",
        "数到3就开始：1、2、3",
        "This sales for 2.5% off, only $12.5.",
        "5G网络是4G网络的升级版，2G网络是3G网络的前身",
        "苹果于2030/1/2发布新 iPhone 2X 系列手机，最低售价仅 ¥12999",
        "这酒...里...有毒...",
        # 异常case
        "只有,,,才是最好的",
        "babala2是什么？",  # babala二是什么?
        "用beta1测试",  # 用beta一测试
        "have you ever been to beta2?",  # have you ever been to beta two?
        "where's the money?",  # where is the money?
        "who's there?",  # who is there?
        "which's the best?",  # which is the best?
        "how's it going?",  # how is it going?
        "今天是个好日子 it's a good day",  # 今天是个好日子 it is a good day
        # 术语
        "such as XTTS, CosyVoice2, Fish-Speech, and F5-TTS",  # such as xtts,cosyvoice two,fish-speech,and f five-tts
        "GPT-5-Nano is the smallest and fastest variant in the GPT-5 model family.",  # GPT-five-Nano is the smallest and fastest variant in the GPT-five model family
        "GPT-5-Nano 是 GPT-5 模型家族中最小且速度最快的变体",  # GPT-五-Nano 是 GPT-五 系统中最小且速度最快的变体
        "2025/09/08 IndexTTS-2 全球发布",  # 二零二五年九月八日 IndexTTS-二全球发布
        "Here are some highly-rated M.2 NVMe SSDs: Samsung 9100 PRO PCIe 5.0 SSD M.2, $139.99",  # Here are some highly-rated M dot two NVMe SSD's, Samsung nine thousand one hundred PRO PCIE five SSD M dot two . one hundred and thirty nine dollars and ninety nine cents
        "we dive deep into the showdown between DisplayPort 1.4 and HDMI 2.1 to determine which is the best choice for gaming enthusiasts",
        # 人名
        "约瑟夫·高登-莱维特（Joseph Gordon-Levitt is an American actor）",
        "蒂莫西·唐纳德·库克（英文名：Timothy Donald Cook），通称蒂姆·库克（Tim Cook），美国商业经理、工业工程师和工业开发商，现任苹果公司首席执行官。",
        # 长句子
        "《盗梦空间》是由美国华纳兄弟影片公司出品的电影，由克里斯托弗·诺兰执导并编剧，莱昂纳多·迪卡普里奥、玛丽昂·歌迪亚、约瑟夫·高登-莱维特、艾利奥特·佩吉、汤姆·哈迪等联袂主演，2010年7月16日在美国上映，2010年9月1日在中国内地上映，2020年8月28日在中国内地重映。影片剧情游走于梦境与现实之间，被定义为“发生在意识结构内的当代动作科幻片”，讲述了由莱昂纳多·迪卡普里奥扮演的造梦师，带领特工团队进入他人梦境，从他人的潜意识中盗取机密，并重塑他人梦境的故事。",
        "清晨拉开窗帘，阳光洒在窗台的Bloomixy花艺礼盒上——薰衣草香薰蜡烛唤醒嗅觉，永生花束折射出晨露般光泽。设计师将“自然绽放美学”融入每个细节：手工陶瓷花瓶可作首饰收纳，香薰精油含依兰依兰舒缓配方。限量款附赠《365天插花灵感手册》，让每个平凡日子都有花开仪式感。\n宴会厅灯光暗下的刹那，Glimmeria星月系列耳坠开始发光——瑞士冷珐琅工艺让蓝宝石如银河流动，钛合金骨架仅3.2g无负重感。设计师秘密：内置微型重力感应器，随步伐产生0.01mm振幅，打造“行走的星光”。七夕限定礼盒含星座定制铭牌，让爱意如星辰永恒闪耀。",
        "电影1：“黑暗骑士”（演员：克里斯蒂安·贝尔、希斯·莱杰；导演：克里斯托弗·诺兰）；电影2：“盗梦空间”（演员：莱昂纳多·迪卡普里奥；导演：克里斯托弗·诺兰）；电影3：“钢琴家”（演员：艾德里安·布洛迪；导演：罗曼·波兰斯基）；电影4：“泰坦尼克号”（演员：莱昂纳多·迪卡普里奥；导演：詹姆斯·卡梅隆）；电影5：“阿凡达”（演员：萨姆·沃辛顿；导演：詹姆斯·卡梅隆）；电影6：“南方公园：大电影”（演员：马特·斯通、托马斯·艾恩格瑞；导演：特雷·帕克）",
    ]
    # 测试分词器
    tokenizer = TextTokenizer(
        vocab_file="checkpoints/bpe.model",
        normalizer=text_normalizer,
    )

    codes = tokenizer.batch_encode(
        cases,
        out_type=int,
    )

    print(f"vocab_size: {tokenizer.vocab_size}")
    # print(f"pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    print(f"bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")
    print(f"eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    print(f"unk_token: {tokenizer.unk_token}, unk_token_id: {tokenizer.unk_token_id}")
    # 测试拼音 (8474-10201)
    for id in range(8474, 10201):
        pinyin = tokenizer.convert_ids_to_tokens(id)
        if re.match(TextNormalizer.PINYIN_TONE_PATTERN, pinyin, re.IGNORECASE) is None:
            print(f"{pinyin} should be matched")
    for badcase in [
        "beta1", "better1", "voice2", "bala2", "babala2", "hunger2"
    ]:
        if re.match(TextNormalizer.PINYIN_TONE_PATTERN, badcase, re.IGNORECASE) is not None:
            print(f"{badcase} should not be matched!")
    # 不应该有 unk_token_id
    for t in set([*TextTokenizer.punctuation_marks_tokens, ",", "▁,", "-", "▁..."]):
        tokens = tokenizer.convert_tokens_to_ids(t)
        if tokenizer.unk_token_id in tokens:
            print(f"Warning: {t} is unknown token")
        print(f"`{t}`", "->", tokens, "->", tokenizer.convert_ids_to_tokens(tokens))
    for ch in set(tokenizer.normalizer.zh_char_rep_map.values()):
        # 测试 normalize后的字符能被分词器识别
        print(f"`{ch}`", "->", tokenizer.sp_model.Encode(ch, out_type=str))
        print(f"` {ch}`", "->", tokenizer.sp_model.Encode(f" {ch}", out_type=str))
    max_text_tokens_per_segment=120
    for i in range(len(cases)):
        print(f"原始文本: {cases[i]}")
        print(f"Normalized: {text_normalizer.normalize(cases[i])}")
        tokens = tokenizer.tokenize(cases[i])
        print("Tokenzied: ", ", ".join([f"`{t}`" for t in tokens]))
        segments = tokenizer.split_segments(tokens, max_text_tokens_per_segment=max_text_tokens_per_segment)
        print("Segments count:", len(segments))
        if len(segments) > 1:
            for j in range(len(segments)):
                print(f"  {j}, count:", len(segments[j]), ", tokens:", "".join(segments[j]))
                if len(segments[j]) > max_text_tokens_per_segment:
                    print(f"Warning: segment {j} is too long, length: {len(segments[j])}")
        #print(f"Token IDs (first 10): {codes[i][:10]}")
        if tokenizer.unk_token in codes[i]:
            print(f"Warning: `{cases[i]}` contains UNKNOWN token")
        print(f"Decoded: {tokenizer.decode(codes[i], do_lower_case=True)}")
        print("-" * 50)
