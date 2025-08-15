# -*- coding: utf-8 -*-
"""
@Author  : API Subtitle Extractor
@Time    : 2024
@FileName: subtitle_extractor_api.py
@desc: API专用的字幕提取器，避免参数冲突
"""
import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path

def extract_subtitles_safe(video_path, subtitle_area=None, language="ch", mode="fast", extract_frequency=3, output_dir=None):
    """
    安全的字幕提取函数，通过独立进程执行，避免参数冲突
    
    Args:
        video_path (str): 视频文件路径
        subtitle_area (tuple): 字幕区域 (y_min, y_max, x_min, x_max)
        language (str): 识别语言
        mode (str): 识别模式
        extract_frequency (int): 提取频率
        output_dir (str): 输出目录，默认为视频文件所在目录
    
    Returns:
        dict: 提取结果
    """
    
    try:
        # 构建命令行参数
        current_dir = os.path.dirname(os.path.abspath(__file__))
        worker_script = os.path.join(current_dir, 'extract_worker.py')
        
        cmd = [
            sys.executable, worker_script,
            '--video-path', video_path,
            '--language', language,
            '--mode', mode,
            '--extract-frequency', str(extract_frequency)
        ]
        
        # 添加字幕区域参数
        if subtitle_area:
            y_min, y_max, x_min, x_max = subtitle_area
            cmd.extend(['--subtitle-area', str(y_min), str(y_max), str(x_min), str(x_max)])
        
        # 添加输出目录参数
        if output_dir:
            cmd.extend(['--output-dir', output_dir])
        else:
            # 默认使用视频文件所在目录
            video_dir = os.path.dirname(os.path.abspath(video_path))
            cmd.extend(['--output-dir', video_dir])
        
        # 执行工作脚本，实时输出到控制台
        from threading import Thread
        
        def stream_output(pipe, prefix=""):
            """实时读取并输出子进程的标准输出"""
            output_lines = []
            try:
                for line in iter(pipe.readline, ''):
                    if line.strip():
                        print(f"{prefix}{line.rstrip()}")
                        output_lines.append(line)
                pipe.close()
            except Exception as e:
                print(f"输出流读取错误: {e}")
            return output_lines
        
        # 启动子进程
        process = subprocess.Popen(
            cmd,
            cwd=current_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 收集输出的列表
        stdout_lines = []
        stderr_lines = []
        
        # 启动输出读取线程
        def collect_stdout():
            nonlocal stdout_lines
            stdout_lines = stream_output(process.stdout, "")
            
        def collect_stderr():
            nonlocal stderr_lines
            stderr_lines = stream_output(process.stderr, "ERR: ")
        
        stdout_thread = Thread(target=collect_stdout)
        stderr_thread = Thread(target=collect_stderr)
        
        stdout_thread.start()
        stderr_thread.start()
        
        # 等待进程完成
        process.wait()
        
        # 等待输出线程完成
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        
        # 创建结果对象
        class Result:
            def __init__(self, returncode, stdout, stderr):
                self.returncode = returncode
                self.stdout = ''.join(stdout)
                self.stderr = ''.join(stderr)
        
        result = Result(process.returncode, stdout_lines, stderr_lines)
        
        if result.returncode == 0:
            # 解析输出结果
            output = result.stdout
            
            # 查找结果标记
            start_marker = "###RESULT_START###"
            end_marker = "###RESULT_END###"
            
            start_index = output.find(start_marker)
            end_index = output.find(end_marker)
            
            if start_index != -1 and end_index != -1:
                json_str = output[start_index + len(start_marker):end_index].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    return {
                        "success": False,
                        "error": f"解析结果JSON失败: {e}",
                        "raw_output": output,
                        "stderr": result.stderr
                    }
            else:
                return {
                    "success": False,
                    "error": "未找到结果标记",
                    "raw_output": output,
                    "stderr": result.stderr
                }
        else:
            return {
                "success": False,
                "error": f"脚本执行失败，返回码: {result.returncode}",
                "stderr": result.stderr,
                "stdout": result.stdout
            }
    
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "字幕提取超时"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"执行异常: {str(e)}"
        }