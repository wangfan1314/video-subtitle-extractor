#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author  : Extract Worker
@Time    : 2024
@FileName: extract_worker.py
@desc: 独立的字幕提取工作进程，避免参数冲突
"""
import os
import sys
import json
import argparse
import multiprocessing

def safe_string(text):
    """安全地处理可能包含特殊字符的字符串"""
    if text is None:
        return ""
    try:
        # 确保是字符串
        if not isinstance(text, str):
            text = str(text)
        # 移除或替换可能导致编码问题的字符
        return text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    except Exception:
        return repr(text)  # 如果还有问题，就返回repr形式

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
from pathlib import Path

# 清除命令行参数，避免PaddleOCR参数冲突，但保留我们自己的参数
original_argv = sys.argv.copy()

class CustomSubtitleExtractor:
    """自定义字幕提取器，支持自定义输出目录"""
    
    def __init__(self, video_path, subtitle_area, output_base_dir, language=None, mode=None):
        self.video_path = video_path
        self.subtitle_area = subtitle_area
        self.output_base_dir = output_base_dir
        self.language = language
        self.mode = mode
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self._video_cap = None  # 缓存VideoCapture对象
        
        # 设置输出目录
        self.temp_output_dir = os.path.join(output_base_dir, f"{self.video_name}_subtitle_output")
        self.frame_output_dir = os.path.join(self.temp_output_dir, 'frames')
        self.subtitle_output_dir = os.path.join(self.temp_output_dir, 'subtitle')
        
        # 创建输出目录
        os.makedirs(self.frame_output_dir, exist_ok=True)
        os.makedirs(self.subtitle_output_dir, exist_ok=True)
        
        # 初始化原始提取器但修改其输出路径
        self._init_extractor()
    
    def _init_extractor(self):
        """初始化原始提取器并修改输出路径"""
        from backend.main import SubtitleExtractor
        
        # 创建原始提取器
        self.extractor = SubtitleExtractor(self.video_path, self.subtitle_area, self.language, self.mode)
        
        # 修改输出路径
        self.extractor.temp_output_dir = self.temp_output_dir
        self.extractor.frame_output_dir = self.frame_output_dir
        self.extractor.subtitle_output_dir = self.subtitle_output_dir
        self.extractor.vsf_subtitle = os.path.join(self.subtitle_output_dir, 'raw_vsf.srt')
        self.extractor.raw_subtitle_path = os.path.join(self.subtitle_output_dir, 'raw.txt')
        
        # 重新创建目录（原始构造函数可能已经创建了默认目录）
        os.makedirs(self.frame_output_dir, exist_ok=True)
        os.makedirs(self.subtitle_output_dir, exist_ok=True)
    
    def frame_to_seconds(self, frame_no):
        """
        将帧号转换为秒数，使用与SRT文件生成相同的方法
        """
        try:
            import cv2
            if self._video_cap is None:
                self._video_cap = cv2.VideoCapture(self.video_path)
            
            # 设置当前帧号
            self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, _ = self._video_cap.read()
            
            if ret:
                milliseconds = self._video_cap.get(cv2.CAP_PROP_POS_MSEC)
                if milliseconds > 0:
                    return round(milliseconds / 1000.0, 2)
            
            # 如果获取失败，使用帧率计算
            return round(frame_no / self.extractor.fps, 2)
        except Exception as e:
            print(f"时间转换失败: {e}", file=sys.stderr)
            # 回退到简单计算
            return round(frame_no / self.extractor.fps, 2)
    
    def __del__(self):
        """析构函数，释放视频资源"""
        if self._video_cap is not None:
            try:
                self._video_cap.release()
            except:
                pass
        
        # 重写需要修复路径的方法
        self.extractor.generate_subtitle_json = self._generate_subtitle_json_custom
        
        # 保存原始方法的引用
        self._original_run = self.extractor.run
        self.extractor.run = self._run_custom
    
    def run(self):
        """运行字幕提取"""
        return self.extractor.run()
    
    @property
    def frame_count(self):
        return self.extractor.frame_count
    
    @property
    def fps(self):
        return self.extractor.fps
    
    @property
    def frame_width(self):
        return self.extractor.frame_width
    
    @property
    def frame_height(self):
        return self.extractor.frame_height
    
    def _generate_subtitle_json_custom(self):
        """
        自定义的JSON生成方法，使用正确的输出目录
        """
        import pysrt
        import json
        
        # 在自定义输出目录中查找SRT文件
        video_basename = os.path.splitext(os.path.basename(self.video_path))[0]
        srt_filename = os.path.join(self.subtitle_output_dir, f"{video_basename}.srt")
        json_filename = os.path.join(os.path.splitext(self.video_path)[0] + '.json')

        print(f"查找SRT文件: {srt_filename}")
        
        # 首先检查原始字幕文件是否存在
        if not os.path.exists(self.extractor.raw_subtitle_path):
            print(f"警告：原始字幕文件不存在: {self.extractor.raw_subtitle_path}")
            # 从已生成的SRT文件中获取字幕数据
            if os.path.exists(srt_filename):
                print(f"从已生成的SRT文件中提取字幕信息: {srt_filename}")
                try:
                    subs = pysrt.open(srt_filename, encoding='utf-8')
                    subtitle_data = []
                    
                    for sub in subs:
                        subtitle_entry = {
                            "index": sub.index,
                            "start_time": str(sub.start),
                            "end_time": str(sub.end),
                            "text": sub.text.strip()
                        }
                        subtitle_data.append(subtitle_entry)
                    
                    # 写入JSON文件
                    with open(json_filename, mode='w', encoding='utf-8') as f:
                        json.dump(subtitle_data, f, ensure_ascii=False, indent=2)
                    
                    print(f"JSON字幕文件已生成: {json_filename}")
                    return
                except Exception as e:
                    print(f"读取SRT文件失败: {e}")
            else:
                print(f"错误：SRT文件也不存在: {srt_filename}")
                # 创建一个空的JSON文件
                empty_subtitle_data = []
                with open(json_filename, mode='w', encoding='utf-8') as f:
                    json.dump(empty_subtitle_data, f, ensure_ascii=False, indent=2)
                print(f"创建空的JSON字幕文件: {json_filename}")
                return
        
        # VSF和非VSF模式都可以使用相同的逻辑
        # 因为_remove_duplicate_subtitle()返回的数据格式在两种模式下是一致的
        mode_info = "VSF模式" if self.extractor.use_vsf else "非VSF模式"
        print(f"{mode_info}：生成JSON文件...")
        
        subtitle_content = self.extractor._remove_duplicate_subtitle()
        
        subtitle_data = []
        for index, content in enumerate(subtitle_content):
            start_frame_no = int(content[0])
            end_frame_no = int(content[1])
            
            # 比较起始帧号与结束帧号， 如果字幕持续时间不足1秒，则将显示时间设为1s
            if abs(end_frame_no - start_frame_no) < self.extractor.fps:
                end_frame_no = start_frame_no + int(self.extractor.fps)
            
            # 计算时间（秒）- 使用与SRT相同的方法
            start_time_sec = self.frame_to_seconds(start_frame_no)
            end_time_sec = self.frame_to_seconds(end_frame_no)
            
            frame_content = content[2]
            
            # 根据语言设置label
            language_label_map = {
                'ch': 'CN',
                'chinese_cht': 'CHT', 
                'en': 'EN',
                'korean': 'KR',
                'japan': 'JP',
                'thai': 'TH',
                'ar': 'AR',
                'es': 'ES',
                'fr': 'FR',
                'de': 'DE',
                'ru': 'RU',
                'pt': 'PT',
                'it': 'IT',
                'vi': 'VI'
            }
            
            # 获取当前语言的label，默认为'CN'
            current_language = getattr(self.extractor, 'actual_language', 'ch')
            label = language_label_map.get(current_language, 'CN')
            
            # 解析坐标信息
            coordinate_str = content[3].strip('()')
            try:
                xmin, xmax, ymin, ymax = map(int, coordinate_str.split(', '))
                
                # 将坐标转换到1000x1000范围内
                # 坐标格式: left=xmin, top=ymin, right=xmax, bottom=ymax
                transformed_left, transformed_top, transformed_right, transformed_bottom = transform_coordinates_to_1000x1000(
                    xmin, ymin, xmax, ymax, 
                    self.extractor.frame_width, 
                    self.extractor.frame_height
                )
                
                # 新的数据结构：rect格式为[left, top, right, bottom]
                subtitle_entry = {
                    "idx": index,  # 从0开始
                    "rect": [transformed_left, transformed_top, transformed_right, transformed_bottom],  # 转换后的坐标
                    "text": frame_content.strip(),
                    "label": label,
                    "start_frame": start_frame_no,
                    "end_frame": end_frame_no,
                    "start_time": start_time_sec,
                    "end_time": end_time_sec
                }
                subtitle_data.append(subtitle_entry)
            except (ValueError, IndexError):
                # 如果坐标解析失败，添加没有位置信息的条目
                subtitle_entry = {
                    "idx": index,  # 从0开始
                    "rect": [0, 0, 0, 0],  # 默认坐标
                    "text": frame_content.strip(),
                    "label": label,
                    "start_frame": start_frame_no,
                    "end_frame": end_frame_no,
                    "start_time": start_time_sec,
                    "end_time": end_time_sec
                }
                subtitle_data.append(subtitle_entry)
        
        # 写入JSON文件到正确的输出目录
        with open(json_filename, mode='w', encoding='utf-8') as f:
            json.dump(subtitle_data, f, ensure_ascii=False, indent=2)
        
        print(f"JSON字幕文件已生成({mode_info}): {json_filename}")
    
    def _run_custom(self):
        """
        自定义的run方法，先执行原始逻辑再处理文件移动
        """
        import backend.config as config
        import shutil
        
        # 确保raw.txt文件存在（在运行前创建）
        if not os.path.exists(self.extractor.raw_subtitle_path):
            os.makedirs(os.path.dirname(self.extractor.raw_subtitle_path), exist_ok=True)
            with open(self.extractor.raw_subtitle_path, 'w', encoding='utf-8') as f:
                f.write("")  # 创建空文件
        
        # 临时禁用缓存删除，避免在文件移动前删除目录
        import sys
        original_debug_no_delete = config.DEBUG_NO_DELETE_CACHE
        config.DEBUG_NO_DELETE_CACHE = True
        
        # 执行原始的run方法
        result = self._original_run()
        
        # 恢复原始设置
        config.DEBUG_NO_DELETE_CACHE = original_debug_no_delete
        
        # 处理文件移动到正确的输出目录
        video_basename = os.path.splitext(os.path.basename(self.video_path))[0]
        
        # 确保输出目录存在
        os.makedirs(self.subtitle_output_dir, exist_ok=True)
        
        # 查找SRT文件的可能位置
        video_dir = os.path.dirname(self.video_path)
        video_basename = os.path.splitext(os.path.basename(self.video_path))[0]
        original_srt_path = os.path.join(os.path.splitext(self.video_path)[0] + '.srt')
        alt_srt_path = os.path.join(video_dir, f"{video_basename}.srt")
        
        # 移动SRT文件
        new_srt_path = os.path.join(self.subtitle_output_dir, f"{video_basename}.srt")
        
        # 尝试多个可能的SRT文件位置
        srt_found = False
        for srt_path in [original_srt_path, alt_srt_path]:
            if os.path.exists(srt_path):
                srt_found = True
                try:
                    shutil.copy2(srt_path, new_srt_path)
                    print(f"SRT文件已复制到: {new_srt_path}")
                    break
                except Exception as e:
                    print(f"复制SRT文件失败: {e}")
        
        if not srt_found:
            print("警告: 未找到SRT文件")
        
        # 移动JSON文件
        original_json_path = os.path.join(os.path.splitext(self.video_path)[0] + '.json')
        alt_json_path = os.path.join(video_dir, f"{video_basename}.json")
        new_json_path = os.path.join(self.subtitle_output_dir, f"{video_basename}.json")
        
        # 尝试多个可能的JSON文件位置
        json_found = False
        for json_path in [original_json_path, alt_json_path]:
            if os.path.exists(json_path):
                json_found = True
                try:
                    shutil.copy2(json_path, new_json_path)
                    print(f"JSON文件已复制到: {new_json_path}")
                    break
                except Exception as e:
                    print(f"复制JSON文件失败: {e}")
        
        if not json_found:
            print("警告: 未找到JSON文件")
        
        # 移动TXT文件（如果存在）
        original_txt_path = os.path.join(os.path.splitext(self.video_path)[0] + '.txt')
        new_txt_path = os.path.join(self.subtitle_output_dir, f"{video_basename}.txt")
        
        if os.path.exists(original_txt_path):
            try:
                shutil.move(original_txt_path, new_txt_path)
                print(f"TXT文件已移动到: {new_txt_path}")
            except Exception as e:
                print(f"移动TXT文件失败: {e}")
        
        # 文件移动完成后，手动清理不需要的缓存文件（但保留我们的输出目录）
        if not original_debug_no_delete:
            try:
                # 只删除frames目录，保留subtitle目录
                frames_dir = os.path.join(self.temp_output_dir, 'frames')
                if os.path.exists(frames_dir):
                    shutil.rmtree(frames_dir, True)
            except Exception as e:
                pass  # 静默处理清理错误
        
        return result

def main():
    # 解析我们自己的参数
    parser = argparse.ArgumentParser(description='字幕提取工作进程')
    parser.add_argument('--video-path', required=True, help='视频文件路径')
    parser.add_argument('--subtitle-area', nargs=4, type=int, help='字幕区域 y_min y_max x_min x_max')
    parser.add_argument('--language', default='ch', help='识别语言')
    parser.add_argument('--mode', default='fast', help='识别模式')
    parser.add_argument('--extract-frequency', type=int, default=3, help='提取频率')
    parser.add_argument('--output-dir', help='输出目录，默认为视频文件所在目录')
    
    args = parser.parse_args(original_argv[1:])
    
    # 清除所有命令行参数，避免PaddleOCR解析
    sys.argv = ['extract_worker.py']
    
    try:
        # 确保在正确的目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # 添加backend路径
        backend_dir = os.path.join(script_dir, 'backend')
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        
        # 导入字幕提取模块
        from backend.main import SubtitleExtractor
        import backend.config as config
        
        # 设置多进程启动方法
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            # 如果已经设置过，忽略错误
            pass
        
        # 更新配置（仅更新全局配置，具体语言参数通过参数传递）
        config.MODE_TYPE = args.mode
        config.EXTRACT_FREQUENCY = args.extract_frequency
        
        # 处理字幕区域
        subtitle_area = None
        if args.subtitle_area:
            y_min, y_max, x_min, x_max = args.subtitle_area
            subtitle_area = (y_min, y_max, x_min, x_max)
        
        # 确定输出目录
        if args.output_dir:
            output_base_dir = args.output_dir
        else:
            # 默认使用视频文件所在的目录
            output_base_dir = os.path.dirname(os.path.abspath(args.video_path))
        
        # 创建自定义的字幕提取器
        extractor = CustomSubtitleExtractor(args.video_path, subtitle_area, output_base_dir, args.language, args.mode)
        
        # 开始提取
        extractor.run()
        
        # 收集结果
        output_dir = extractor.subtitle_output_dir
        subtitles = []
        output_files = {}
        
        # 查找生成的字幕文件
        for ext in ['srt', 'json', 'txt']:
            for file_path in Path(output_dir).glob(f"*.{ext}"):
                key = f"{ext}_{file_path.stem}"
                output_files[key] = str(file_path)
        
        # 尝试读取JSON格式的字幕文件
        json_files = list(Path(output_dir).glob("*.json"))
        if json_files:
            try:
                with open(json_files[0], 'r', encoding='utf-8') as f:
                    subtitle_data = json.load(f)
                    # 直接使用新的JSON格式，确保字符串安全
                    subtitles = []
                    for item in subtitle_data:
                        safe_item = {}
                        for key, value in item.items():
                            if isinstance(value, str):
                                safe_item[key] = safe_string(value)
                            else:
                                safe_item[key] = value
                        subtitles.append(safe_item)
            except Exception as e:
                print(f"读取JSON字幕文件失败: {e}", file=sys.stderr)
        
        # 输出结果，确保所有字符串都是安全的
        result = {
            "success": True,
            "subtitles": subtitles,
            "output_files": {k: safe_string(v) for k, v in output_files.items()},
            "video_info": {
                "path": safe_string(args.video_path),
                "frame_count": extractor.frame_count,
                "fps": extractor.fps,
                "width": extractor.frame_width,
                "height": extractor.frame_height
            }
        }
        
        # 输出结果到标准输出，确保编码正确
        print("###RESULT_START###")
        try:
            # 在Windows上使用ensure_ascii=True避免编码问题
            print(json.dumps(result, ensure_ascii=True))
        except UnicodeEncodeError:
            # 如果仍有编码问题，则转换为UTF-8字节然后解码
            result_json = json.dumps(result, ensure_ascii=True)
            print(result_json)
        print("###RESULT_END###")
        
    except Exception as e:
        # 输出错误结果
        error_result = {
            "success": False,
            "error": safe_string(str(e))
        }
        print("###RESULT_START###")
        try:
            # 在Windows上使用ensure_ascii=True避免编码问题
            print(json.dumps(error_result, ensure_ascii=True))
        except UnicodeEncodeError:
            # 如果仍有编码问题，则转换为UTF-8字节然后解码
            error_json = json.dumps(error_result, ensure_ascii=True)
            print(error_json)
        print("###RESULT_END###")
        
        # 也输出到stderr用于调试
        import traceback
        try:
            print(f"错误详情: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
        except UnicodeEncodeError:
            # 如果stderr也有编码问题，则使用ASCII安全的方式
            print(f"错误详情: {repr(str(e))}", file=sys.stderr)
            print("traceback信息包含非ASCII字符，已跳过", file=sys.stderr)

if __name__ == "__main__":
    main()