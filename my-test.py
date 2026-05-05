from ultralytics import YOLO
import os

# 核心：获取 train.py 所在的 code 目录。由绝对路径获得文件夹路径。
code_folder = os.path.dirname(os.path.abspath(__file__))

# 1. 加载你训练好的模型
model = YOLO("../results/bridge_yolov11/weights/best.pt")

# 2. 验证测试集（关键：指定 split="test"）
metrics = model.val(
    data=os.path.join(code_folder, "../dataset/data.yaml"),
    split="test"
    save=True,         # 生成带检测框的图片
    save_txt=True,     # 保存检测结果txt
    imgsz=640,          # 图片尺寸（和训练一致）
    device="cpu",       # 指定CPU/GPU
    conf=0.25,          # 建议加：置信度阈值
    iou=0.6,            # 建议加：NMS去重阈值
    
    project=os.path.join(code_folder,"../results"),  # 自定义输出根目录
    name="test_result",                         # 文件夹名称
    plots=True          # 生成PR曲线、混淆矩阵
)

# 3. 输出核心评分指标
print("mAP50:", metrics.box.map50)      # 最关键指标，综合精度
print("mAP50-95:", metrics.box.map)     # 更严格的综合精度
print("精确率 Precision:", metrics.box.p)# 预测框中正确的比例（误检率）
print("召回率 Recall:", metrics.box.r)  # 真实目标中被检测到的比例（漏检率）