from ultralytics import YOLO
import os

# 核心：获取 train.py 所在的 code 目录。由绝对路径获得文件夹路径。
code_folder = os.path.dirname(os.path.abspath(__file__))

# 加载你训练好的模型
model = YOLO("../results/bridge_yolov11/weights/best.pt")

# 预测（关键：指定 project 和 name）
model.predict(
    source=os.path.join(code_folder,"../dataset/test/images"),       # 你的测试图片
    save=True,                 # 保存画框图
    save_txt=True,             # 保存检测结果
    imgsz=640,
    device="cpu",
    project=os.path.join(code_folder,"../results"),    # 输出根目录
    name="detection_result"     # 输出文件夹名称
)