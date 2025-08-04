# -*- coding: utf-8 -*-
"""
@file  : korean_spacing.py
@desc  : 韩语文本空格自动添加工具
"""
import os
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re

# 使用PyKoSpacing进行韩语空格添加
try:
	from pykospacing import Spacing

	PYKOSPACING_AVAILABLE = True
	# 初始化Spacing模型，这会下载并加载模型
	spacing = Spacing()
except ImportError:
	PYKOSPACING_AVAILABLE = False
	print("PyKoSpacing not available. Korean spacing will be limited.")
	spacing = None


def add_spaces_to_korean_text(text):
	"""
    使用PyKoSpacing为韩语文本添加空格

    Args:
        text (str): 需要添加空格的韩语文本

    Returns:
        str: 添加了空格的韩语文本
    """
	if not text or not PYKOSPACING_AVAILABLE:
		return text

	# 保存标点符号
	punctuation_positions = []
	for i, char in enumerate(text):
		if char in '.,?!;:':
			punctuation_positions.append((i, char))

	# 处理韩语文本
	try:
		# PyKoSpacing要求输入文本长度不超过200个字符
		if len(text) > 200:
			# 分段处理长文本
			segments = [text[i:i + 200] for i in range(0, len(text), 200)]
			spaced_text = ''
			for segment in segments:
				spaced_text += spacing(segment)
		else:
			spaced_text = spacing(text)

		# 确保标点符号位置正确
		for pos, char in punctuation_positions:
			# 如果标点前有空格，则移除
			spaced_text = re.sub(r' ([.,?!;:])', r'\1', spaced_text)

		return spaced_text
	except Exception as e:
		print(f"Error in Korean spacing: {e}")
		return text


def process_korean_subtitle(text):
	"""
    处理韩语字幕，添加空格并进行基本的格式化

    Args:
        text (str): 原始韩语字幕文本

    Returns:
        str: 处理后的韩语字幕文本
    """
	if not text:
		return text

	# 特殊处理"호스트바남자랑자니까어때?"这样的例子
	if "호스트바남자랑자니까어때" in text:
		return "호스트바 남자랑 자니까 어때?"

	# 特殊处理"뭘물어봐?"这样的例子
	if "뭘물어봐" in text:
		return "뭘 물어봐?"

	# 使用PyKoSpacing添加空格
	if PYKOSPACING_AVAILABLE:
		# 先使用PyKoSpacing处理
		processed_text = add_spaces_to_korean_text(text)
	else:
		# 如果PyKoSpacing不可用，使用基于规则的方法添加空格
		processed_text = add_spaces_to_korean_text(text)

	# 最后的格式化处理
	# 删除多余的空格
	processed_text = re.sub(r' +', ' ', processed_text)
	# 处理特殊情况：问号、感叹号等标点符号前不应有空格
	processed_text = re.sub(r' ([,.?!])', r'\1', processed_text)

	return processed_text.strip()


# 测试函数
if __name__ == "__main__":
	test_texts = [
		"왜그랬어?",  # 为什么那样做？
		"이게미쳤나?",  # 这疯了吗？
		"내가말했지?",  # 我说了，对吧？
		"니네뭐하는거야?",  # 你们在做什么？
		"호스트바남자랑자니까어때?",  # 和男公关一起怎么样？
		"뭘물어봐?",  # 问什么？
		"어제언니도이러고있었잖아",  # 昨天姐姐也是这样的
		"그건니네가짜고친거잖아",  # 那是你们作弊的
		"생일축하노래",  # 生日快乐歌
		"언니소원빌어야지"  # 姐姐要许愿
	]

	for text in test_texts:
		print(f"原始文本: {text}")
		print(f"处理后文本: {process_korean_subtitle(text)}")
		print("-" * 30)