#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author  : Realtime Client
@Time    : 2024
@FileName: realtime_client.py
@desc: 实时进度监控客户端
"""

import requests
import time
import json
import sys
from datetime import datetime

class RealtimeSubtitleClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def submit_task(self, video_path, subtitle_area=None, language="ch", mode="fast", output_dir=None):
        """提交字幕提取任务"""
        url = f"{self.base_url}/extract"
        
        payload = {
            "video_path": video_path,
            "language": language,
            "mode": mode,
            "extract_frequency": 3
        }
        
        if subtitle_area:
            payload["subtitle_area"] = subtitle_area
        
        if output_dir:
            payload["output_dir"] = output_dir
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"提交任务失败: {e}")
            return None
    
    def monitor_task_progress(self, task_id, poll_interval=2):
        """实时监控任务进度"""
        print(f"🔍 开始监控任务: {task_id}")
        print("=" * 60)
        
        last_messages = []
        start_time = time.time()
        
        while True:
            try:
                # 获取任务状态
                response = requests.get(f"{self.base_url}/status/{task_id}")
                response.raise_for_status()
                status_data = response.json()
                
                # 显示基本状态
                current_time = datetime.now().strftime("%H:%M:%S")
                elapsed = int(time.time() - start_time)
                
                print(f"\r[{current_time}] 运行时间: {elapsed}s | 状态: {status_data['status']} | {status_data['message'][:50]}...", end="", flush=True)
                
                # 显示新的进度消息
                progress_messages = status_data.get('progress_messages', [])
                if progress_messages:
                    new_messages = progress_messages[len(last_messages):]
                    for msg in new_messages:
                        print(f"\n📋 {msg}")
                    last_messages = progress_messages
                
                # 显示最新输出
                latest_output = status_data.get('latest_output')
                if latest_output and latest_output not in [msg for msg in last_messages]:
                    print(f"\n💬 {latest_output}")
                
                # 检查任务状态
                if status_data['status'] == 'completed':
                    print(f"\n\n✅ 任务完成！")
                    self._display_results(status_data)
                    break
                elif status_data['status'] == 'failed':
                    print(f"\n\n❌ 任务失败: {status_data['message']}")
                    break
                
                time.sleep(poll_interval)
                
            except requests.exceptions.RequestException as e:
                print(f"\n❌ 获取状态失败: {e}")
                break
            except KeyboardInterrupt:
                print(f"\n\n⏹️ 监控已停止")
                break
    
    def _display_results(self, status_data):
        """显示任务结果"""
        if 'subtitles' in status_data and status_data['subtitles']:
            subtitles = status_data['subtitles']
            print(f"\n📝 提取到 {len(subtitles)} 条字幕:")
            
            # 显示前5条字幕
            for i, subtitle in enumerate(subtitles[:5]):
                print(f"  {i+1}. {subtitle['start_time']} --> {subtitle['end_time']}")
                print(f"     {subtitle['text']}")
            
            if len(subtitles) > 5:
                print(f"  ... 还有 {len(subtitles) - 5} 条字幕")
        
        if 'output_files' in status_data and status_data['output_files']:
            print(f"\n📁 输出文件:")
            for file_type, file_path in status_data['output_files'].items():
                print(f"  • {file_type}: {file_path}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='实时字幕提取监控客户端')
    parser.add_argument('--server', default='http://localhost:8000', help='API服务地址')
    parser.add_argument('--video', required=True, help='视频文件路径')
    parser.add_argument('--area', nargs=4, type=int, metavar=('Y_MIN', 'Y_MAX', 'X_MIN', 'X_MAX'),
                       help='字幕区域坐标')
    parser.add_argument('--language', default='ch', help='识别语言')
    parser.add_argument('--mode', default='fast', choices=['fast', 'accurate', 'auto'], help='识别模式')
    parser.add_argument('--output', help='输出目录')
    parser.add_argument('--poll', type=int, default=2, help='状态查询间隔（秒）')
    
    args = parser.parse_args()
    
    client = RealtimeSubtitleClient(args.server)
    
    print("🚀 实时字幕提取监控客户端")
    print("=" * 60)
    print(f"视频文件: {args.video}")
    print(f"字幕区域: {args.area}")
    print(f"识别语言: {args.language}")
    print(f"识别模式: {args.mode}")
    print(f"输出目录: {args.output or '视频文件所在目录'}")
    print("=" * 60)
    
    # 提交任务
    print("📤 提交字幕提取任务...")
    result = client.submit_task(
        video_path=args.video,
        subtitle_area=args.area,
        language=args.language,
        mode=args.mode,
        output_dir=args.output
    )
    
    if not result:
        print("❌ 任务提交失败")
        return
    
    task_id = result['task_id']
    print(f"✅ 任务已提交，ID: {task_id}")
    
    # 开始监控
    client.monitor_task_progress(task_id, args.poll)

if __name__ == "__main__":
    main()