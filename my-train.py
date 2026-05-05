# 从ultralytics库中导入YOLO类，用来创建、训练、预测模型
from ultralytics import YOLO

import os
# 核心：获取 train.py 所在的 code 目录。由绝对路径获得文件夹路径。
code_folder = os.path.dirname(os.path.abspath(__file__))

# 加载你自己放在 ../models/ 里的 yolo11n.pt 预训练模型
# 不从网上下载，直接用本地文件
model = YOLO("../models/yolo11n.pt")

# 程序入口（Windows 多进程训练必须写这个，防止报错）
if __name__ == "__main__":

    # 开始训练
    model.train(
        # data="dataset/data.yaml",    # 指定数据集配置文件（类别、路径等）
        data=os.path.join(code_folder, "../dataset/data.yaml"),

        epochs=1,                       # 训练 5 轮（完整看一遍所有数据）
        imgsz=640,                      # 把图片统一缩放到 640x640 大小训练
        batch=2,                        # 一次喂 2 张图（CPU 必须小，否则卡死）
        device="cpu",                   # 用 CPU 训练
        
        # project="../results",           # 训练结果保存到 code/runs/detect/results/...
        
        project=os.path.join(code_folder, "../results"),  # 结果保存到上一级（proj_GYU-DET/results）

        name="bridge_yolov11",          # 结果文件夹名叫 bridge_yolov11
        save=True,                      # 自动保存最好的模型 best.pt
        verbose=True                    # 打印详细训练日志
    )