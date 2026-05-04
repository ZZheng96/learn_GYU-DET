
"""
桥梁病害检测程序
使用YOLOv11 nano模型进行桥梁病害检测
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime

# 病害类别名称
CLASS_NAMES = ['Crack', 'Breakage', 'Comb', 'Hole', 'Reinforcement', 'Seepage']

# 每个类别对应的颜色
CLASS_COLORS = {
    'Crack': (255, 0, 0),          # 红色
    'Breakage': (0, 255, 0),       # 绿色
    'Comb': (0, 0, 255),           # 蓝色
    'Hole': (255, 255, 0),         # 青色
    'Reinforcement': (255, 0, 255), # 洋红色
    'Seepage': (0, 255, 255)       # 黄色
}

def load_model(model_path):
    """
    加载YOLOv11 nano模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    model = YOLO(model_path)
    return model

def detect_single_image(model, image_path, conf_threshold=0.25, iou_threshold=0.45):
    """
    对单张图片进行病害检测
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    # 运行推理
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )

    return results[0]

def visualize_detection(image, result, save_path=None):
    """
    可视化检测结果
    """
    img = image.copy()

    # 获取检测框
    boxes = result.boxes

    for box in boxes:
        # 获取坐标和类别信息
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        class_id = int(box.cls[0].cpu().numpy())
        confidence = float(box.conf[0].cpu().numpy())

        class_name = CLASS_NAMES[class_id]
        color = CLASS_COLORS[class_name]

        # 绘制检测框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 绘制类别标签和置信度
        label = f"{class_name}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        # 标签背景
        cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)

        # 标签文本
        cv2.putText(img, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if save_path:
        cv2.imwrite(save_path, img)

    return img

def detect_directory(model, input_dir, output_dir, conf_threshold=0.25, iou_threshold=0.45):
    """
    对目录中的所有图片进行病害检测
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # 统计信息
    total_images = 0
    total_detections = 0

    # 遍历输入目录
    for root, _, files in os.walk(input_dir):
        for file in files:
            # 检查文件扩展名
            if Path(file).suffix.lower() in image_extensions:
                image_path = os.path.join(root, file)

                try:
                    # 读取图片
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"无法读取图片: {image_path}")
                        continue

                    # 进行检测
                    result = detect_single_image(model, image_path, conf_threshold, iou_threshold)

                    # 统计检测数量
                    num_detections = len(result.boxes)
                    total_detections += num_detections
                    total_images += 1

                    # 生成输出路径
                    relative_path = os.path.relpath(image_path, input_dir)
                    output_path = os.path.join(output_dir, relative_path)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    # 可视化并保存结果
                    visualize_detection(image, result, output_path)

                    print(f"处理完成: {relative_path} | 检测到 {num_detections} 个病害")

                except Exception as e:
                    print(f"处理图片 {image_path} 时出错: {str(e)}")

    print(f"\n检测完成！共处理 {total_images} 张图片，检测到 {total_detections} 个病害")

def main():
    parser = argparse.ArgumentParser(description='桥梁病害检测程序')
    parser.add_argument('--model', type=str, default='weights/yolov11n.pt',
                       help='模型权重文件路径')
    parser.add_argument('--source', type=str, required=True,
                       help='输入图片或目录路径')
    parser.add_argument('--output', type=str, default='results',
                       help='输出目录路径')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值 (0-1)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IOU阈值 (0-1)')

    args = parser.parse_args()

    # 加载模型
    print(f"正在加载模型: {args.model}")
    model = load_model(args.model)
    print("模型加载成功！")

    # 判断输入是单张图片还是目录
    if os.path.isfile(args.source):
        # 单张图片检测
        print(f"正在检测图片: {args.source}")
        image = cv2.imread(args.source)
        if image is None:
            print("无法读取图片！")
            return

        result = detect_single_image(model, args.source, args.conf, args.iou)
        num_detections = len(result.boxes)
        print(f"检测完成！检测到 {num_detections} 个病害")

        # 保存结果
        os.makedirs(args.output, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output, f"detection_{timestamp}.jpg")
        visualize_detection(image, result, output_path)
        print(f"结果已保存到: {output_path}")

    elif os.path.isdir(args.source):
        # 目录批量检测
        print(f"正在检测目录: {args.source}")
        detect_directory(model, args.source, args.output, args.conf, args.iou)
    else:
        print("输入路径不存在！")

if __name__ == '__main__':
    main()
