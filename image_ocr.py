# -*- coding: utf-8 -*-
"""
@Author  : Image OCR
@Time    : 2024
@FileName: image_ocr.py
@desc: 图片OCR识别模块
"""
import os
import sys
import requests
import tempfile
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import urllib.parse
import mimetypes

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.append(str(Path(__file__).parent))

# 延迟导入，避免在模块级别导入可能缺失的依赖
def _import_ocr_modules():
    """延迟导入OCR相关模块"""
    try:
        from backend.tools.ocr import OcrRecogniser, get_coordinates
        return OcrRecogniser, get_coordinates
    except ImportError as e:
        raise ImportError(f"无法导入OCR模块: {str(e)}。请确保所有依赖已正确安装。")


class ImageOCRProcessor:
    """图片OCR处理器"""
    
    def __init__(self, language: str = "ch", confidence_threshold: float = 0.5):
        """
        初始化图片OCR处理器
        
        Args:
            language: 识别语言，默认为中文
            confidence_threshold: 置信度阈值，低于此值的结果将被过滤
        """
        self.language = language
        self.confidence_threshold = confidence_threshold
        self.ocr_recognizer = None
        
    def _init_ocr_recognizer(self):
        """延迟初始化OCR识别器"""
        if self.ocr_recognizer is None:
            OcrRecogniser, _ = _import_ocr_modules()
            self.ocr_recognizer = OcrRecogniser(language=self.language, mode="fast")
    
    def download_image(self, image_url: str) -> str:
        """
        下载图片到本地临时文件
        
        Args:
            image_url: 图片URL
            
        Returns:
            str: 本地临时文件路径
            
        Raises:
            Exception: 下载失败时抛出异常
        """
        try:
            # 验证URL格式
            parsed_url = urllib.parse.urlparse(image_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError("无效的图片URL格式")
            
            # 设置请求头，模拟浏览器访问
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # 下载图片
            response = requests.get(image_url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            
            # 检查内容类型
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                # 尝试从URL推断文件类型
                guessed_type, _ = mimetypes.guess_type(image_url)
                if not guessed_type or not guessed_type.startswith('image/'):
                    raise ValueError(f"URL返回的不是图片文件，Content-Type: {content_type}")
            
            # 确定文件扩展名
            file_extension = '.jpg'  # 默认扩展名
            if 'png' in content_type.lower():
                file_extension = '.png'
            elif 'gif' in content_type.lower():
                file_extension = '.gif'
            elif 'webp' in content_type.lower():
                file_extension = '.webp'
            elif 'bmp' in content_type.lower():
                file_extension = '.bmp'
            
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            
            # 写入图片数据
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            
            temp_file.close()
            
            # 验证下载的文件是否为有效图片
            try:
                with Image.open(temp_file.name) as img:
                    img.verify()  # 验证图片完整性
            except Exception as e:
                os.unlink(temp_file.name)  # 删除无效文件
                raise ValueError(f"下载的文件不是有效的图片: {str(e)}")
            
            return temp_file.name
            
        except requests.exceptions.Timeout:
            raise Exception("图片下载超时")
        except requests.exceptions.ConnectionError:
            raise Exception("网络连接错误，无法下载图片")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP错误: {e.response.status_code}")
        except Exception as e:
            raise Exception(f"图片下载失败: {str(e)}")
    
    def get_image_resolution(self, image_path: str) -> Tuple[int, int]:
        """
        获取图片分辨率
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            Tuple[int, int]: (width, height)
        """
        try:
            with Image.open(image_path) as img:
                return img.size  # PIL返回的是(width, height)
        except Exception as e:
            raise Exception(f"获取图片分辨率失败: {str(e)}")
    
    def recognize_text(self, image_path: str) -> List[Dict[str, Any]]:
        """
        识别图片中的文字
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            List[Dict]: 识别结果列表，每个元素包含文字内容、坐标和置信度
        """
        try:
            # 初始化OCR识别器
            self._init_ocr_recognizer()
            
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("无法读取图片文件")
            
            # 获取图片尺寸
            height, width = image.shape[:2]
            
            # 使用OCR识别器进行识别
            dt_boxes, rec_results = self.ocr_recognizer.predict(image)
            
            # 获取坐标信息
            _, get_coordinates = _import_ocr_modules()
            coordinates = get_coordinates(dt_boxes)
            
            # 处理识别结果
            results = []
            for i, (text_info, coordinate) in enumerate(zip(rec_results, coordinates)):
                text, confidence = text_info
                
                # 过滤低置信度结果
                if confidence < self.confidence_threshold:
                    continue
                
                # 提取坐标信息 (xmin, xmax, ymin, ymax)
                xmin, xmax, ymin, ymax = coordinate
                
                result = {
                    "text": text.strip(),
                    "confidence": float(confidence),
                    "bbox": {
                        "left": int(xmin),
                        "top": int(ymin),
                        "right": int(xmax),
                        "bottom": int(ymax),
                        "width": int(xmax - xmin),
                        "height": int(ymax - ymin)
                    },
                    "position": {
                        "x": int((xmin + xmax) / 2),  # 中心点x坐标
                        "y": int((ymin + ymax) / 2)   # 中心点y坐标
                    }
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            raise Exception(f"文字识别失败: {str(e)}")
    
    def process_image_url(self, image_url: str) -> Dict[str, Any]:
        """
        处理图片URL，完成下载、识别等完整流程
        
        Args:
            image_url: 图片URL
            
        Returns:
            Dict: 包含识别结果和图片信息的字典
        """
        temp_image_path = None
        try:
            # 下载图片
            temp_image_path = self.download_image(image_url)
            
            # 获取图片分辨率
            width, height = self.get_image_resolution(temp_image_path)
            
            # 识别文字
            ocr_results = self.recognize_text(temp_image_path)
            
            # 提取所有识别到的文字
            all_text = " ".join([result["text"] for result in ocr_results])
            
            return {
                "success": True,
                "image_info": {
                    "url": image_url,
                    "width": width,
                    "height": height,
                    "local_path": temp_image_path
                },
                "ocr_results": ocr_results,
                "text_count": len(ocr_results),
                "all_text": all_text.strip(),
                "language": self.language,
                "confidence_threshold": self.confidence_threshold
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "image_url": image_url
            }
        finally:
            # 清理临时文件
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.unlink(temp_image_path)
                except:
                    pass  # 忽略删除失败的错误


def recognize_image_text(image_url: str, language: str = "ch", confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """
    便捷函数：识别图片中的文字
    
    Args:
        image_url: 图片URL
        language: 识别语言
        confidence_threshold: 置信度阈值
        
    Returns:
        Dict: 识别结果
    """
    processor = ImageOCRProcessor(language=language, confidence_threshold=confidence_threshold)
    return processor.process_image_url(image_url)


if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description='图片OCR识别测试')
    parser.add_argument('--image-url', required=True, help='图片URL')
    parser.add_argument('--language', default='ch', help='识别语言')
    parser.add_argument('--confidence', type=float, default=0.5, help='置信度阈值')
    
    args = parser.parse_args()
    
    result = recognize_image_text(args.image_url, args.language, args.confidence)
    
    if result["success"]:
        print(f"图片尺寸: {result['image_info']['width']}x{result['image_info']['height']}")
        print(f"识别到 {result['text_count']} 个文字区域")
        print(f"全部文字: {result['all_text']}")
        print("\n详细结果:")
        for i, ocr_result in enumerate(result["ocr_results"], 1):
            print(f"{i}. 文字: {ocr_result['text']}")
            print(f"   置信度: {ocr_result['confidence']:.3f}")
            print(f"   位置: ({ocr_result['bbox']['left']}, {ocr_result['bbox']['top']}) - ({ocr_result['bbox']['right']}, {ocr_result['bbox']['bottom']})")
    else:
        print(f"识别失败: {result['error']}")
