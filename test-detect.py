from ultralytics import YOLO

# 加载你训练好的模型
model = YOLO("../results/bridge_yolov11/weights/best.pt")

# 预测（关键：指定 project 和 name）
model.predict(
    source="../images/test/images",       # 你的测试图片
    save=True,                 # 保存画框图
    save_txt=True,             # 保存检测结果
    imgsz=640,
    device="cpu",
    project="../results",    # 输出根目录
    name="bridge_result"     # 输出文件夹名称
)