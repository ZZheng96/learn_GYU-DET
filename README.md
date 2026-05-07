
```
├── code
├── dataset
│   ├── classes.txt
│   ├── data.yaml
│   ├── test
│   │   ├── images
│   │   └── labels
│   ├── train
│   │   ├── images
│   │   └── labels
│   └── val
│        ├── images
│        └── labels
├── models
│   └── yolo11n.pt
└── results
     ├── bridge_yolov11
     │   └── weights
     │       ├── best.pt
     │       └── last.pt
     └── test_result
```


data.yaml
```
# 桥梁病害检测数据集配置文件

# 数据集路径（实际不是相对于此文件的位置，而是相对工作目录code的位置）
path: ..  # 数据集根目录
train: dataset/train/images  # 训练集图片路径
val: dataset/val/images      # 验证集图片路径
test: dataset/test/images     # 测试集图片路径

# 类别数量
nc: 6

# 类别名称
names:
  0: Crack
  1: Breakage
  2: Comb
  3: Hole
  4: Reinforcement
  5: Seepage
```