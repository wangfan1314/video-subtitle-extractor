#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author  : Progress Monitor
@Time    : 2024
@FileName: progress_monitor.py
@desc: 实时进度监控器
"""

import threading
import time
import sys
from io import StringIO
import contextlib

class ProgressCapture:
    """捕获和转发进度输出"""
    
    def __init__(self, task_id, task_manager):
        self.task_id = task_id
        self.task_manager = task_manager
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.captured_output = []
        
    def write(self, text):
        """重写write方法以捕获输出"""
        # 实时输出到控制台（保持原有的输出效果）
        self.original_stdout.write(text)
        self.original_stdout.flush()
        
        # 保存输出用于API返回
        self.captured_output.append(text)
        
        # 更新任务状态
        self._update_progress_from_text(text)
        
        return len(text)
    
    def flush(self):
        """flush方法"""
        self.original_stdout.flush()
    
    def _is_progress_info(self, text):
        """判断是否是进度信息"""
        progress_keywords = [
            "进度", "完成", "%", "帧", "识别", "提取", "处理中",
            "frame", "progress", "OCR", "extract"
        ]
        return any(keyword in text for keyword in progress_keywords)
    
    def _update_progress_from_text(self, text):
        """从文本中提取进度信息并更新任务状态"""
        try:
            # 更新任务消息，包含最新的进度信息
            clean_text = text.strip()
            if clean_text and self.task_id:
                current_task = self.task_manager.get_task(self.task_id)
                if current_task:
                    # 保留最近的进度消息
                    if 'progress_messages' not in current_task:
                        current_task['progress_messages'] = []
                    
                    # 只记录有意义的消息，避免重复
                    if clean_text and clean_text not in current_task['progress_messages'][-3:]:
                        current_task['progress_messages'].append(clean_text)
                        # 只保留最近的15条消息
                        if len(current_task['progress_messages']) > 15:
                            current_task['progress_messages'] = current_task['progress_messages'][-15:]
                    
                    # 更新主要消息为最新的进度信息
                    if any(keyword in clean_text for keyword in ['处理中', '结束', '开始', '完成', '%', '进度']):
                        self.task_manager.update_task(
                            self.task_id, 
                            message=clean_text,
                            latest_output=clean_text
                        )
        except Exception as e:
            # 静默处理错误，避免影响主流程
            pass

class ProgressMonitor:
    """进度监控器"""
    
    def __init__(self, task_id, task_manager):
        self.task_id = task_id
        self.task_manager = task_manager
        self.capture = ProgressCapture(task_id, task_manager)
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        # 重定向标准输出
        sys.stdout = self.capture
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        # 恢复标准输出
        sys.stdout = self.capture.original_stdout
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 定期检查任务状态
                time.sleep(1)
                
                # 可以在这里添加其他监控逻辑
                task = self.task_manager.get_task(self.task_id)
                if task and task.get('status') in ['completed', 'failed']:
                    self.monitoring = False
                    break
                    
            except Exception as e:
                print(f"监控循环错误: {e}")
                break

def create_progress_monitor(task_id, task_manager):
    """创建进度监控器"""
    return ProgressMonitor(task_id, task_manager)

@contextlib.contextmanager
def capture_progress(task_id, task_manager):
    """上下文管理器，用于捕获进度"""
    monitor = create_progress_monitor(task_id, task_manager)
    try:
        monitor.start_monitoring()
        yield monitor
    finally:
        monitor.stop_monitoring()