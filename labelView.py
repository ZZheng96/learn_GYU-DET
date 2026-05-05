import cv2
import os
from tkinter import Tk, filedialog

# UI选择文件夹
root = Tk()
root.withdraw()
root.attributes("-topmost", True)

print("请选择 图片文件夹")
IMG_DIR = filedialog.askdirectory(title="选择图片文件夹")
print("请选择 labels 标注文件夹")
LABEL_DIR = filedialog.askdirectory(title="选择标注文件夹")

# 支持图片格式
img_suffix = (".jpg", ".jpeg", ".png", ".bmp")

# 显示最大宽高（可自行调）
DISP_MAX_W = 1000
DISP_MAX_H = 900

# 框线粗细、字体大小
BOX_THICKNESS = 15      # 框加粗
FONT_SCALE = 5
FONT_THICKNESS = 10

for img_name in os.listdir(IMG_DIR):
    if not img_name.lower().endswith(img_suffix):
        continue

    img_path = os.path.join(IMG_DIR, img_name)
    txt_name = os.path.splitext(img_name)[0] + ".txt"
    txt_path = os.path.join(LABEL_DIR, txt_name)

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    # 画标注框
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or len(line.split()) != 5:
                continue
            cls, cx, cy, bw, bh = map(float, line.split())
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)

            # 加粗画框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), BOX_THICKNESS)
            # 加类别文字，更好辨认
            cv2.putText(img, f"cls:{int(cls)}", (x1, y1-8), 
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 
                        (0, 255, 0), FONT_THICKNESS)

    # 等比例缩小显示图片
    scale = min(DISP_MAX_W / w, DISP_MAX_H / h)
    new_size = (int(w * scale), int(h * scale))
    img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    cv2.imshow(f"Preview | {img_name}", img_resized)
    cv2.waitKey(0)

cv2.destroyAllWindows()