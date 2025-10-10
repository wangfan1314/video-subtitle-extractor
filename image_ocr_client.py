#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author  : Image OCR Client
@Time    : 2024
@FileName: image_ocr_client.py
@desc: 图片OCR识别客户端示例
"""
import requests
import json
import argparse
import sys


def recognize_image_text(base_url, image_url, language="ch", confidence_threshold=0.5):
    """
    调用API识别图片中的文字
    
    Args:
        base_url: API服务地址
        image_url: 图片URL
        language: 识别语言
        confidence_threshold: 置信度阈值
        
    Returns:
        dict: 识别结果
    """
    try:
        # 构建请求数据
        request_data = {
            "image_url": image_url,
            "language": language,
            "confidence_threshold": confidence_threshold
        }
        
        print(f"🔍 正在识别图片: {image_url}")
        print(f"🌍 识别语言: {language}")
        print(f"📊 置信度阈值: {confidence_threshold}")
        print("⏳ 请稍候...")
        
        # 发送请求
        response = requests.post(
            f"{base_url}/recognize-image",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "error": f"HTTP错误: {response.status_code}",
                "details": response.text
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": f"无法连接到API服务 {base_url}，请确保服务已启动"
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "请求超时"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"请求异常: {str(e)}"
        }


def print_results(result):
    """
    打印识别结果
    
    Args:
        result: 识别结果字典
    """
    print("\n" + "=" * 60)
    
    if result.get("success"):
        print("✅ 图片文字识别成功!")
        
        # 打印图片信息
        image_info = result.get("image_info", {})
        if image_info:
            print(f"\n📏 图片信息:")
            print(f"   URL: {image_info.get('url', 'N/A')}")
            print(f"   尺寸: {image_info.get('width', 'N/A')}x{image_info.get('height', 'N/A')}")
        
        # 打印识别统计
        print(f"\n📊 识别统计:")
        print(f"   识别语言: {result.get('language', 'N/A')}")
        print(f"   置信度阈值: {result.get('confidence_threshold', 'N/A')}")
        print(f"   文字区域数量: {result.get('text_count', 0)}")
        
        # 打印全部文字
        all_text = result.get('all_text', '').strip()
        if all_text:
            print(f"\n📄 识别到的全部文字:")
            print(f"   {all_text}")
        else:
            print(f"\n⚠️ 未识别到任何文字")
        
        # 打印详细结果
        ocr_results = result.get('ocr_results', [])
        if ocr_results:
            print(f"\n📝 详细识别结果:")
            for i, ocr_result in enumerate(ocr_results, 1):
                print(f"\n   {i}. 文字: '{ocr_result['text']}'")
                print(f"      置信度: {ocr_result['confidence']:.3f}")
                
                bbox = ocr_result['bbox']
                print(f"      边界框: 左上({bbox['left']}, {bbox['top']}) - 右下({bbox['right']}, {bbox['bottom']})")
                print(f"      尺寸: {bbox['width']}x{bbox['height']}")
                
                position = ocr_result['position']
                print(f"      中心点: ({position['x']}, {position['y']})")
        
    else:
        print("❌ 图片文字识别失败!")
        error = result.get("error", "未知错误")
        print(f"   错误信息: {error}")
        
        # 如果有详细错误信息，也打印出来
        details = result.get("details")
        if details:
            print(f"   详细信息: {details}")
    
    print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='图片OCR识别客户端',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 识别中文图片
  python image_ocr_client.py --image-url "https://example.com/chinese_image.jpg"
  
  # 识别英文图片
  python image_ocr_client.py --image-url "https://example.com/english_image.jpg" --language en
  
  # 设置置信度阈值
  python image_ocr_client.py --image-url "https://example.com/image.jpg" --confidence 0.7
  
  # 使用自定义API地址
  python image_ocr_client.py --base-url "http://192.168.1.100:8000" --image-url "https://example.com/image.jpg"

支持的语言:
  ch - 简体中文 (默认)
  chinese_cht - 繁体中文
  en - 英文
  japan - 日文
  korean - 韩文
  thai - 泰文
  ar - 阿拉伯文
  更多语言请参考API文档
        """
    )
    
    parser.add_argument('--base-url', default='http://localhost:8000', 
                       help='API服务地址 (默认: http://localhost:8000)')
    parser.add_argument('--image-url', required=True, 
                       help='图片URL (必需)')
    parser.add_argument('--language', default='ch', 
                       help='识别语言 (默认: ch)')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='置信度阈值，0-1之间 (默认: 0.5)')
    parser.add_argument('--output-json', 
                       help='将结果保存为JSON文件')
    
    args = parser.parse_args()
    
    # 验证置信度阈值
    if not 0 <= args.confidence <= 1:
        print("❌ 置信度阈值必须在0-1之间")
        sys.exit(1)
    
    # 调用API识别图片
    result = recognize_image_text(
        base_url=args.base_url,
        image_url=args.image_url,
        language=args.language,
        confidence_threshold=args.confidence
    )
    
    # 打印结果
    print_results(result)
    
    # 保存JSON结果（如果指定了输出文件）
    if args.output_json:
        try:
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n💾 结果已保存到: {args.output_json}")
        except Exception as e:
            print(f"\n❌ 保存JSON文件失败: {str(e)}")
    
    # 设置退出码
    if result.get("success"):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
