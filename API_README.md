# 视频字幕提取API服务

这个项目已经从命令行工具升级为API服务，现在可以通过HTTP接口调用字幕提取功能。

## 快速开始

### 1. 环境准备

首先激活虚拟环境：
```bash
source /Users/wangzhizhong/workspace/virturlEnv/videoEnv/bin/activate
```

安装API相关依赖：
```bash
pip install -r requirements.txt
```

### 2. 启动服务

#### 方式一：使用脚本启动（推荐）
```bash
./start_server.sh
```

#### 方式二：使用Python启动
```bash
python start_service.py
```

#### 方式三：自定义参数启动
```bash
python start_service.py --host 0.0.0.0 --port 8080 --workers 2
```

服务启动后，你可以通过以下地址访问：
- API服务：http://localhost:8000
- API文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

## API接口说明

### 1. 提取字幕
**POST** `/extract`

提交视频字幕提取任务

#### 请求参数
```json
{
  "video_path": "/path/to/video.mp4",
  "subtitle_area": [1350, 1600, 0, 1080],
  "extract_frequency": 3,
  "language": "ch",
  "mode": "fast",
  "output_dir": "/path/to/output"
}
```

参数说明：
- `video_path` (必需): 视频文件的完整路径
- `subtitle_area` (可选): 字幕区域坐标 [y_min, y_max, x_min, x_max]
- `extract_frequency` (可选): 每秒提取帧数，默认为3
- `language` (可选): 识别语言，默认为"ch"（中文）
- `mode` (可选): 识别模式，可选值：fast（快速）、accurate（精确）、auto（自动）
- `output_dir` (可选): 输出目录，默认为视频文件所在目录

#### 响应
```json
{
  "task_id": "uuid-string",
  "status": "processing",
  "message": "任务已提交，正在处理中",
  "progress": {
    "total": 0,
    "frame_extract": 0,
    "ocr": 0,
    "finished": false
  }
}
```

### 2. 查询任务状态
**GET** `/status/{task_id}`

查询指定任务的处理状态

#### 响应
```json
{
  "task_id": "uuid-string",
  "status": "completed",
  "message": "字幕提取完成",
  "subtitles": [
    {
      "index": 1,
      "start_time": "00:00:01,000",
      "end_time": "00:00:03,000",
      "text": "提取的字幕文本",
      "position": {
        "left": 100,
        "right": 900,
        "top": 1350,
        "bottom": 1600
      }
    }
  ],
  "output_files": {
    "srt_output": "/path/to/output.srt",
    "json_output": "/path/to/output.json"
  }
}
```

### 3. 列出所有任务
**GET** `/tasks`

获取所有任务的列表

### 4. 删除任务
**DELETE** `/tasks/{task_id}`

删除指定的任务记录

### 5. 图片OCR识别
**POST** `/recognize-image`

识别图片中的文字内容

#### 请求参数
```json
{
  "image_url": "https://example.com/image.jpg",
  "language": "ch",
  "confidence_threshold": 0.5
}
```

参数说明：
- `image_url` (必需): 图片URL地址，支持HTTP/HTTPS
- `language` (可选): 识别语言，默认为"ch"（中文）
- `confidence_threshold` (可选): 置信度阈值，0-1之间，默认为0.5

#### 响应
```json
{
  "success": true,
  "message": "图片文字识别成功",
  "image_info": {
    "url": "https://example.com/image.jpg",
    "width": 800,
    "height": 400
  },
  "ocr_results": [
    {
      "text": "识别到的文字",
      "confidence": 0.95,
      "bbox": {
        "left": 100,
        "top": 50,
        "right": 200,
        "bottom": 80,
        "width": 100,
        "height": 30
      },
      "position": {
        "x": 150,
        "y": 65
      }
    }
  ],
  "text_count": 1,
  "all_text": "识别到的文字",
  "language": "ch",
  "confidence_threshold": 0.5
}
```

## 客户端调用示例

### Python客户端

#### 视频字幕提取
使用提供的客户端脚本：
```bash
python client_example.py --video /path/to/video.mp4 --area 1350 1600 0 1080 --language ch --mode fast
```

#### 图片OCR识别
使用图片OCR客户端脚本：
```bash
# 识别中文图片
python image_ocr_client.py --image-url "https://example.com/chinese_image.jpg"

# 识别英文图片
python image_ocr_client.py --image-url "https://example.com/english_image.jpg" --language en

# 设置置信度阈值
python image_ocr_client.py --image-url "https://example.com/image.jpg" --confidence 0.7

# 保存结果到JSON文件
python image_ocr_client.py --image-url "https://example.com/image.jpg" --output-json result.json
```

### curl调用示例
```bash
# 提交任务（输出到视频文件所在目录）
curl -X POST "http://localhost:8000/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/Users/wangzhizhong/Documents/202508041842.mp4",
    "subtitle_area": [1350, 1600, 0, 1080],
    "language": "ch",
    "mode": "fast"
  }'

# 提交任务（自定义输出目录）
curl -X POST "http://localhost:8000/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/Users/wangzhizhong/Documents/202508041842.mp4",
    "subtitle_area": [1350, 1600, 0, 1080],
    "language": "ch",
    "mode": "fast",
    "output_dir": "/Users/wangzhizhong/Desktop"
  }'

# 查询任务状态
curl "http://localhost:8000/status/your-task-id"

# 健康检查
curl "http://localhost:8000/health"

# 图片OCR识别
curl -X POST "http://localhost:8000/recognize-image" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "language": "ch",
    "confidence_threshold": 0.5
  }'
```

### JavaScript/前端调用示例
```javascript
// 提交字幕提取任务
const extractSubtitles = async (videoPath, subtitleArea) => {
  const response = await fetch('http://localhost:8000/extract', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      video_path: videoPath,
      subtitle_area: subtitleArea,
      language: 'ch',
      mode: 'fast'
    })
  });
  
  const result = await response.json();
  return result.task_id;
};

// 查询任务状态
const checkStatus = async (taskId) => {
  const response = await fetch(`http://localhost:8000/status/${taskId}`);
  return await response.json();
};

// 使用示例
const taskId = await extractSubtitles('/path/to/video.mp4', [1350, 1600, 0, 1080]);
console.log('任务ID:', taskId);

// 轮询检查状态
const pollStatus = async (taskId) => {
  const status = await checkStatus(taskId);
  console.log('任务状态:', status);
  
  if (status.status === 'processing') {
    setTimeout(() => pollStatus(taskId), 5000); // 5秒后再次检查
  }
};

pollStatus(taskId);

// 图片OCR识别
const recognizeImageText = async (imageUrl, language = 'ch', confidenceThreshold = 0.5) => {
  const response = await fetch('http://localhost:8000/recognize-image', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image_url: imageUrl,
      language: language,
      confidence_threshold: confidenceThreshold
    })
  });
  
  return await response.json();
};

// 使用示例
const ocrResult = await recognizeImageText('https://example.com/image.jpg', 'ch', 0.5);
if (ocrResult.success) {
  console.log('识别成功:', ocrResult.all_text);
  console.log('详细结果:', ocrResult.ocr_results);
} else {
  console.error('识别失败:', ocrResult.error);
}
```

## 支持的语言

- `ch`: 简体中文
- `chinese_cht`: 繁体中文
- `en`: 英文
- `japan`: 日文
- `korean`: 韩文
- `ar`: 阿拉伯文
- `thai`: 泰文
- 更多语言请参考config.py中的配置

## 注意事项

1. **文件路径**: 确保视频文件路径正确，且服务有读取权限
2. **字幕区域**: 字幕区域坐标格式为 [y_min, y_max, x_min, x_max]，如果不指定将使用默认检测
3. **并发限制**: 默认最多同时处理2个任务，避免资源占用过多
4. **输出文件**: 
   - 默认保存在视频文件所在目录下的 `{视频名称}_subtitle_output/` 文件夹
   - 可通过 `output_dir` 参数自定义输出目录
   - 输出文件包括：SRT字幕文件、JSON格式字幕、原始文本等
5. **虚拟环境**: 确保在正确的虚拟环境中运行服务

## 输出目录说明

### 默认输出目录
如果不指定 `output_dir` 参数，输出文件将保存在视频文件所在的目录：
```
/Users/wangzhizhong/Documents/202508041842.mp4
→ 输出到: /Users/wangzhizhong/Documents/202508041842_subtitle_output/
```

### 自定义输出目录
指定 `output_dir` 参数后，输出文件将保存在指定目录：
```json
{
  "video_path": "/Users/wangzhizhong/Documents/202508041842.mp4",
  "output_dir": "/Users/wangzhizhong/Desktop"
}
→ 输出到: /Users/wangzhizhong/Desktop/202508041842_subtitle_output/
```

### 输出文件结构
```
{视频名称}_subtitle_output/
├── frames/          # 提取的视频帧
├── subtitle/        # 字幕文件
│   ├── *.srt       # SRT字幕文件
│   ├── *.json      # JSON格式字幕
│   └── *.txt       # 原始文本
```

## 故障排除

### 服务无法启动
1. 检查虚拟环境是否正确激活
2. 确认所有依赖已安装：`pip install -r requirements.txt`
3. 检查端口8000是否被占用
4. 如果出现参数冲突错误，请使用最新版本的API服务

### 字幕提取失败
1. 确认视频文件路径正确，且服务有读取权限
2. 检查视频文件格式是否支持（MP4、FLV、AVI等）
3. 验证字幕区域坐标是否合理
4. 查看服务日志获取详细错误信息
5. 确保有足够的磁盘空间存储临时文件

### 测试API功能
运行测试脚本验证功能：
```bash
python test_extract_api.py
```

### 测试工作脚本
直接测试字幕提取：
```bash
python extract_worker.py --video-path './test/test_cn.mp4' --language ch --mode fast
```

### 性能优化
1. 使用GPU模式（如果可用）
2. 调整 `extract_frequency` 参数
3. 选择合适的识别模式（fast vs accurate）
4. 精确指定字幕区域减少检测范围

## 从命令行工具迁移

如果你之前使用命令行方式：
```bash
python backend/main.py
```

现在可以改为API调用：
```bash
# 启动服务
./start_server.sh

# 在另一个终端调用API
python client_example.py --video /path/to/video.mp4 --area 1350 1600 0 1080
```

这样就完成了从工具到服务的转换！