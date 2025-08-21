from enum import Enum


# 默认字幕出现的大致区域
class SubtitleArea(Enum):
    # 字幕区域出现在下半部分
    LOWER_PART = 0
    # 字幕区域出现在上半部分
    UPPER_PART = 1
    # 不知道字幕区域可能出现的位置
    UNKNOWN = 2
    # 明确知道字幕区域出现的位置
    CUSTOM = 3


class BackgroundColor(Enum):
    # 字幕背景
    WHITE = 0
    DARK = 1
    UNKNOWN = 2


BGR_COLOR_GREEN = (0, 0xff, 0)
BGR_COLOR_BLUE = (0xff, 0, 0)
BGR_COLOR_RED = (0, 0, 0xff)
BGR_COLOR_WHITE = (0xff, 0xff, 0xff)


def transform_coordinates_to_1000x1000(left, top, right, bottom, video_width, video_height):
    """
    将原始视频坐标转换到1000x1000范围内
    
    Args:
        left, top, right, bottom: 原始坐标
        video_width, video_height: 视频分辨率
    
    Returns:
        tuple: (new_left, new_top, new_right, new_bottom)
    
    转换规则:
    - left和top向下取整 (使用int())
    - right和bottom向上取整 (使用math.ceil())
    """
    import math
    
    # 计算缩放比例
    scale_x = 1000.0 / video_width
    scale_y = 1000.0 / video_height
    
    # 转换坐标
    new_left = int(left * scale_x)          # 向下取整
    new_top = int(top * scale_y)            # 向下取整
    new_right = math.ceil(right * scale_x)  # 向上取整
    new_bottom = math.ceil(bottom * scale_y) # 向上取整
    
    # 确保坐标在有效范围内
    new_left = max(0, min(999, new_left))
    new_top = max(0, min(999, new_top))
    new_right = max(1, min(1000, new_right))
    new_bottom = max(1, min(1000, new_bottom))
    
    # 确保right > left 和 bottom > top
    if new_right <= new_left:
        new_right = new_left + 1
    if new_bottom <= new_top:
        new_bottom = new_top + 1
    
    return new_left, new_top, new_right, new_bottom