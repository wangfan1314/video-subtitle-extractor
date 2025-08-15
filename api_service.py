# -*- coding: utf-8 -*-
"""
@Author  : API Service
@Time    : 2024
@FileName: api_service.py
@desc: 视频字幕提取API服务
"""
import os
import sys
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import threading
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor
import uuid

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.append(str(Path(__file__).parent))

# 创建FastAPI应用
app = FastAPI(
    title="视频字幕提取API",
    description="提供视频硬编码字幕提取服务",
    version="1.0.0"
)

# 请求模型
class SubtitleExtractionRequest(BaseModel):
    video_path: str
    subtitle_area: Optional[List[int]] = None  # [y_min, y_max, x_min, x_max]
    extract_frequency: Optional[int] = 3  # 每秒提取帧数
    language: Optional[str] = "ch"  # 识别语言
    mode: Optional[str] = "fast"  # 识别模式: fast, accurate, auto
    output_dir: Optional[str] = None  # 输出目录，默认为视频文件所在目录
    
    class Config:
        schema_extra = {
            "example": {
                "video_path": "/path/to/video.mp4",
                "subtitle_area": [1350, 1600, 0, 1080],
                "extract_frequency": 3,
                "language": "ch",
                "mode": "fast",
                "output_dir": "/path/to/output"
            }
        }

# 响应模型
class SubtitleResult(BaseModel):
    index: int
    start_time: str
    end_time: str
    text: str
    position: Optional[Dict[str, int]] = None

class SubtitleExtractionResponse(BaseModel):
    task_id: str
    status: str  # "processing", "completed", "failed"
    message: str
    subtitles: Optional[List[SubtitleResult]] = None
    output_files: Optional[Dict[str, str]] = None
    progress: Optional[Dict[str, Any]] = None

# 任务状态管理
class TaskManager:
    def __init__(self):
        self.tasks = {}
        self.executor = ThreadPoolExecutor(max_workers=2)  # 限制并发任务数
    
    def create_task(self, task_id: str):
        """创建新任务"""
        self.tasks[task_id] = {
            "status": "processing",
            "message": "任务已创建，开始处理",
            "created_at": time.time(),
            "progress": {
                "total": 0,
                "frame_extract": 0,
                "ocr": 0,
                "finished": False
            },
            "result": None,
            "error": None
        }
    
    def update_task(self, task_id: str, **kwargs):
        """更新任务状态"""
        if task_id in self.tasks:
            self.tasks[task_id].update(kwargs)
    
    def get_task(self, task_id: str):
        """获取任务状态"""
        return self.tasks.get(task_id)
    
    def submit_task(self, task_id: str, func, *args, **kwargs):
        """提交任务到线程池"""
        future = self.executor.submit(func, task_id, *args, **kwargs)
        return future

# 全局任务管理器
task_manager = TaskManager()

def extract_subtitles_task(task_id: str, request_data: SubtitleExtractionRequest):
    """字幕提取任务（在后台线程中执行）"""
    try:
        # 导入安全的字幕提取器和进度监控
        from subtitle_extractor_api import extract_subtitles_safe
        from progress_monitor import capture_progress
        
        # 验证视频文件是否存在
        if not os.path.exists(request_data.video_path):
            raise FileNotFoundError(f"视频文件不存在: {request_data.video_path}")
        
        # 处理字幕区域参数
        subtitle_area = None
        if request_data.subtitle_area and len(request_data.subtitle_area) == 4:
            y_min, y_max, x_min, x_max = request_data.subtitle_area
            subtitle_area = (y_min, y_max, x_min, x_max)
        
        task_manager.update_task(task_id, message="开始提取视频字幕，启动进度监控")
        
        # 控制台输出任务信息
        print("=" * 80)
        print(f"🎬 新的字幕提取任务开始")
        print(f"📋 任务ID: {task_id}")
        print(f"🎥 视频文件: {request_data.video_path}")
        print(f"📁 输出目录: {request_data.output_dir or '视频文件所在目录'}")
        print(f"📍 字幕区域: {subtitle_area}")
        print(f"🌍 识别语言: {request_data.language}")
        print(f"⚡ 处理模式: {request_data.mode}")
        print(f"📊 提取频率: {request_data.extract_frequency or 3}/秒")
        print("=" * 80)
        
        # 使用进度监控执行字幕提取
        with capture_progress(task_id, task_manager):
            # 调用安全的字幕提取函数
            extract_result = extract_subtitles_safe(
                video_path=request_data.video_path,
                subtitle_area=subtitle_area,
                language=request_data.language or "ch",
                mode=request_data.mode or "fast",
                extract_frequency=request_data.extract_frequency or 3,
                output_dir=request_data.output_dir
            )
        
        if extract_result.get("success"):
            # 提取成功
            subtitles = []
            for item in extract_result.get("subtitles", []):
                subtitles.append(SubtitleResult(
                    index=item.get("index", 0),
                    start_time=item.get("start_time", ""),
                    end_time=item.get("end_time", ""),
                    text=item.get("text", ""),
                    position=item.get("position")
                ))
            
            # 构建响应结果
            result = {
                "subtitles": subtitles,
                "output_files": extract_result.get("output_files", {}),
                "video_info": extract_result.get("video_info", {})
            }
            
            # 控制台输出完成信息
            print("=" * 80)
            print(f"✅ 任务完成! ID: {task_id}")
            print(f"📝 提取到字幕条数: {len(result.get('subtitles', []))}")
            print(f"📁 输出文件数量: {len(result.get('output_files', {}))}")
            if result.get('output_files'):
                print("📋 生成的文件:")
                for file_type, file_path in result['output_files'].items():
                    print(f"   • {file_type}: {file_path}")
            print("=" * 80)
            
            # 更新任务状态为完成
            task_manager.update_task(
                task_id,
                status="completed",
                message="字幕提取完成",
                result=result,
                progress={"total": 100, "frame_extract": 100, "ocr": 100, "finished": True}
            )
        else:
            # 提取失败
            error_msg = extract_result.get("error", "未知错误")
            raise Exception(error_msg)
        
    except Exception as e:
        # 控制台输出失败信息
        print("=" * 80)
        print(f"❌ 任务失败! ID: {task_id}")
        print(f"📋 失败原因: {str(e)}")
        print(f"🎥 视频文件: {request_data.video_path}")
        print("=" * 80)
        
        # 更新任务状态为失败
        task_manager.update_task(
            task_id,
            status="failed",
            message=f"字幕提取失败: {str(e)}",
            error=str(e)
        )

@app.post("/extract", response_model=SubtitleExtractionResponse)
async def extract_subtitles(request: SubtitleExtractionRequest, background_tasks: BackgroundTasks):
    """
    提取视频字幕
    """
    # 生成唯一的任务ID
    task_id = str(uuid.uuid4())
    
    # 创建任务
    task_manager.create_task(task_id)
    
    # 在后台启动字幕提取任务
    task_manager.submit_task(task_id, extract_subtitles_task, request)
    
    return SubtitleExtractionResponse(
        task_id=task_id,
        status="processing",
        message="任务已提交，正在处理中",
        progress={"total": 0, "frame_extract": 0, "ocr": 0, "finished": False}
    )

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    获取任务状态（包含实时进度信息）
    """
    task = task_manager.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    response_data = {
        "task_id": task_id,
        "status": task["status"],
        "message": task["message"],
        "progress": task["progress"],
        "created_at": task.get("created_at"),
        "latest_output": task.get("latest_output"),
        "progress_messages": task.get("progress_messages", [])
    }
    
    # 如果任务完成，添加结果
    if task["status"] == "completed" and task["result"]:
        response_data["subtitles"] = task["result"]["subtitles"]
        response_data["output_files"] = task["result"]["output_files"]
    
    return response_data

@app.get("/tasks")
async def list_tasks():
    """
    列出所有任务
    """
    tasks = []
    for task_id, task_info in task_manager.tasks.items():
        tasks.append({
            "task_id": task_id,
            "status": task_info["status"],
            "message": task_info["message"],
            "created_at": task_info["created_at"],
            "progress": task_info["progress"]
        })
    
    return {"tasks": tasks}

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """
    删除任务记录
    """
    if task_id in task_manager.tasks:
        del task_manager.tasks[task_id]
        return {"message": f"任务 {task_id} 已删除"}
    else:
        raise HTTPException(status_code=404, detail="任务不存在")

@app.get("/")
async def root():
    """
    API根路径
    """
    return {
        "message": "视频字幕提取API服务",
        "version": "1.0.0",
        "endpoints": {
            "extract": "POST /extract - 提取视频字幕",
            "status": "GET /status/{task_id} - 获取任务状态",
            "tasks": "GET /tasks - 列出所有任务",
            "delete": "DELETE /tasks/{task_id} - 删除任务"
        }
    }

@app.get("/health")
async def health_check():
    """
    健康检查
    """
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    # 启动服务
    uvicorn.run(
        "api_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )