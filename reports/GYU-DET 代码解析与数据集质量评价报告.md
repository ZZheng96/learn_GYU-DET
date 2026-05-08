

> 论文来源：Ruiping Li et al., "Multi-defect type beam bridge dataset: GYU-DET", Scientific Data (2025) 12:1101.  
DOI: [10.1038/s41597-025-05395-w](https://doi.org/10.1038/s41597-025-05395-w)  
代码仓库：[https://github.com/IamSunday/Multi-defect-type-beam-bridge-dataset-GYU-DET](https://github.com/IamSunday/Multi-defect-type-beam-bridge-dataset-GYU-DET)
>
> 数据下载地址: [https://doi.org/10.57760/sciencedb.19893](https://doi.org/10.57760/sciencedb.19893)
>

---

## 一、项目概述
GYU-DET 是一个专为**梁式桥梁表面病害检测**任务设计的大规模公开数据集，由贵阳大学等单位联合构建。数据集包含 **11,123 张高分辨率图像**，涵盖裂缝、剥落、蜂窝面、露筋、渗水、孔洞共 **6 类病害**，采用 YOLO 格式进行标注。配套代码以 YOLOv11 为检测骨干，提供了完整的训练脚本和标注格式转换工具。

---

## 二、代码文件结构总览
```plain
source/
├── README.md               # 项目说明文件，包含安装、使用说明和依赖列表
├── txt2coco.py             # 核心工具：YOLO .txt 格式 → COCO .json 格式转换
├── txt2xml.py              # 核心工具：YOLO .txt 格式 → VOC .xml 格式转换
└── ultralytics/
    ├── GZ-DET.yaml         # 数据集配置文件，定义路径和类别
    ├── mytrain.py          # 训练入口脚本，加载模型并启动训练
    ├── cfg/
    │   ├── models/11/yolo11.yaml   # YOLOv11 网络结构定义
    │   ├── default.yaml            # ultralytics 默认超参数配置
    │   └── datasets/               # 通用数据集配置（COCO/VOC 等，非项目专属）
    ├── data/               # 数据加载与增强模块（ultralytics 框架依赖）
    ├── engine/             # 训练/验证/推理引擎（ultralytics 框架依赖）
    ├── models/yolo/detect/ # 检测任务的训练、验证、预测模块（框架依赖）
    ├── nn/                 # 神经网络模块（激活函数、卷积、注意力等）
    └── utils/              # 通用工具函数（指标计算、损失函数等）
```

> **说明**：`ultralytics/` 目录下的绝大部分代码来自 [ultralytics 官方库 v8.3.33](https://github.com/ultralytics/ultralytics)，属于**框架依赖代码**，不在本次注释范围内。项目**自有核心代码**为：`txt2coco.py`、`txt2xml.py`、`ultralytics/mytrain.py`、`ultralytics/GZ-DET.yaml`。
>

---

## 三、核心代码文件工作原理详解
### 3.1 `txt2coco.py` — YOLO 格式转 COCO JSON 格式
#### 功能定位
将 YOLO 格式（`.txt`）的标注文件批量转换为 COCO 格式（`.json`），使数据集能够兼容使用 COCO API 进行评测的检测框架（如 Detectron2、MMDetection 等）。

#### 数据流向
```plain
dataset/
├── images/         ← 原始图像 (.jpg/.png)
├── labels/         ← YOLO 格式标注 (.txt，每行: class_id cx cy w h)
└── classes.txt     ← 类别名称列表（每行一个类别）
        ↓
  txt2coco.py
        ↓
dataset/annotations/
├── train.json      ← COCO 格式训练集标注
├── val.json        ← COCO 格式验证集标注
└── test.json       ← COCO 格式测试集标注（或单个 valid.json）
```

#### 坐标转换逻辑
YOLO 使用**归一化中心点+宽高**格式；COCO 使用**绝对像素左上角+宽高**格式：

| 格式 | 字段 | 描述 |
| --- | --- | --- |
| YOLO | `cx, cy, w, h` | 相对图像尺寸归一化（0~1） |
| COCO | `x1, y1, w_px, h_px` | 绝对像素坐标（左上角+宽高） |


转换公式：

```plain
x1 = (cx - w/2) * W_image
y1 = (cy - h/2) * H_image
w_px = w * W_image
h_px = h * H_image
```

#### 支持三种模式
| 参数 | 模式 | 说明 |
| --- | --- | --- |
| 无参数 | 全量模式 | 所有图像输出到单个 JSON |
| `--random_split` | 随机划分 | 按 8:1:1 自动划分三个 JSON |
| `--split_by_file` | 文件划分 | 读取 train.txt/val.txt/test.txt 按列表划分 |


---

### 3.2 `txt2xml.py` — YOLO 格式转 Pascal VOC XML 格式
#### 功能定位
将 YOLO 格式标注转换为 Pascal VOC 格式（`.xml`），兼容以 VOC 格式为基础的训练框架（如早期 Faster R-CNN 实现、PaddleDetection 等）。

#### 输出 XML 结构
```xml
<annotation>
  <folder>driving_annotation_dataset</folder>
  <filename>1.jpg</filename>
  <size>
    <width>4608</width>
    <height>3456</height>
    <depth>3</depth>
  </size>
  <object>
    <name>Crack</name>       <!-- 类别名（通过 ID→名称 字典映射） -->
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>1024</xmin>      <!-- 左上角 x（绝对像素，1-indexed） -->
      <ymin>768</ymin>
      <xmax>1280</xmax>      <!-- 右下角 x（绝对像素，1-indexed） -->
      <ymax>960</ymax>
    </bndbox>
  </object>
</annotation>

```

#### 坐标转换逻辑
VOC 使用 **1-indexed** 绝对像素坐标（从 1 开始计数），因此在还原时额外加 1：

```plain
xmin = int(cx_norm * W + 1 - w_norm * 0.5 * W)
ymin = int(cy_norm * H + 1 - h_norm * 0.5 * H)
xmax = int(cx_norm * W + 1 + w_norm * 0.5 * W)
ymax = int(cy_norm * H + 1 + h_norm * 0.5 * H)
```

#### 注意事项
+ 当前版本 `dic` 字典中有 **7 个类别**（0~6），但数据集实际为 6 类（0~5），`'6': "Seepage"` 的 ID 与 YAML 配置中的映射存在不一致（YAML 中 Seepage 为 5），使用时需核对 `classes.txt` 中的顺序。

---

### 3.3 `ultralytics/mytrain.py` — 模型训练入口
#### 功能定位
调用 ultralytics 框架，以 YOLOv11n 为基础模型结构，在 GYU-DET 数据集上从头开始训练。

#### 训练流程
```plain
mytrain.py
    │
    ├─ YOLO("yolo11.yaml")       ← 按配置文件初始化 YOLOv11n 网络结构（随机权重）
    │
    └─ model.train(
           data='./GZ-DET.yaml', ← 数据集配置（路径 + 类别）
           epochs=300            ← 训练轮次（论文设置）
       )
           │
           ├─ 每 epoch: 前向传播 → 计算损失(box/cls/dfl) → 反向传播 → 权重更新
           ├─ 验证: 在 val 集上计算 Precision/Recall/mAP@0.5
           └─ 输出: runs/detect/train/ 目录（权重文件、训练曲线、混淆矩阵等）
```

#### 训练代码
```plain
from ultralytics import YOLO  
# 从 ultralytics 包导入 YOLO 类，该类封装了 YOLOv11 模型的训练、推理、导出等全部接口

if __name__ == '__main__':
    # Windows 多进程安全入口：DataLoader 在 Windows 下使用 spawn 方式创建子进程，
    # 训练代码必须放在 if __name__ == '__main__' 块内，否则会因子进程递归导入而报错。

    # 通过 YOLO 配置文件初始化模型结构（从头随机初始化权重，不加载预训练权重）
    # "yolo11.yaml" 是 YOLO11n 模型的网络结构配置文件，定义了各层的类型、通道数、深度因子等超参数
    model = YOLO("yolo11.yaml")

    # 启动模型训练
    # data='./GZ-DET.yaml'  : 数据集配置文件，指定训练集/验证集路径及类别名称
    # epochs=300            : 训练轮次，论文中使用 300 轮以充分收敛
    # 其余超参数（学习率、batch_size、图像尺寸等）均使用 ultralytics 默认值
    # （论文中 batch_size=16，其他参数默认）
    results = model.train(data='./GZ-DET.yaml', epochs=300)
    # results 为 ultralytics 返回的训练结果对象，包含各 epoch 的 loss、mAP 等指标
```

#### 论文实验配置
| 配置项 | 值 |
| --- | --- |
| 模型 | YOLOv11n（最小模型，319层，9.5MB） |
| Epochs | 300 |
| Batch Size | 16 |
| 硬件 | NVIDIA GeForce RTX 4090 |
| CUDA | 12.1 |
| Python | 3.11 |
| 框架 | PyTorch |


---

### 3.4 `ultralytics/GZ-DET.yaml` — 数据集配置文件
#### 功能定位
向 ultralytics 框架描述数据集的路径布局和类别映射，是连接数据与模型的"桥梁"。

#### 类别定义对照表
| ID | 名称 | 中文名 | 说明 |
| --- | --- | --- | --- |
| 0 | Crack | 裂缝 | 混凝土表面线状损伤 |
| 1 | Breakage | 剥落/破损 | 混凝土表层脱落 |
| 2 | Comb | 蜂窝面 | 混凝土表面局部空洞 |
| 3 | Hole | 孔洞 | 较大孔隙，影响承载力 |
| 4 | Reinforcement | 露筋 | 钢筋裸露于表面 |
| 5 | Seepage | 渗水 | 水渍或潮湿斑 |


#### yaml代码
```plain
# GYU-DET 数据集配置文件（供 YOLOv11 训练使用）
# 论文数据集全名：Multi-defect type beam bridge dataset: GYU-DET
# 发布于 Scientific Data (2025) 12:1101, DOI: 10.1038/s41597-025-05395-w
# 数据下载地址: https://doi.org/10.57760/sciencedb.19893

# --- 数据路径配置 ---
# 注意：路径为相对于 ultralytics 包根目录的相对路径，或写为绝对路径
train: /train    # 训练集图像目录（内含 images/ 和 labels/ 子目录），约 8,898 张（8:1:1 划分）
val:   /valid    # 验证集图像目录，约 1,113 张，训练期间每 epoch 结束后自动评估

# --- 类别定义 ---
# GYU-DET 数据集包含 6 类桥梁表面病害，按中国《公路桥梁技术状况评定标准》分类
# 类别 ID（0~5）与 YOLO .txt 标注文件中第一列数字一一对应
names:
  0: Crack          # 裂缝：混凝土表面的线状损伤，影响结构强度和耐久性
  1: Breakage       # 剥落/破损（Spalling）：混凝土表面或内部逐渐松动、脱落的现象
  2: Comb           # 蜂窝面（Honeycomb Surface）：混凝土表面粗糙不平，局部存在空洞
  3: Hole           # 孔洞（Holes）：混凝土结构中较大的空隙，显著影响承载能力
  4: Reinforcement  # 露筋（Exposed Rebar）：混凝土保护层破坏导致钢筋裸露于表面
  5: Seepage        # 渗水（Seepage）：水通过裂缝或孔隙进入结构并在表面形成水渍

# 数据集统计（来自论文）：
#   总图像数：11,123 张 RGB JPG 图像，主分辨率 4608×3456
#   正样本：10,432 张（含至少一个缺陷标注）
#   负样本：691 张（无缺陷，作为对照样本）
#   训练/验证/测试划分比例：8:1:1
#   类别分布不均衡：Breakage 和 Reinforcement 实例数远多于 Crack 和 Hole

```

---

## 四、代码整体数据流
```plain
原始数据集（YOLO 格式）
    images/1.jpg ... labels/1.txt ...
         │
         ├──────────────────┬───────────────────
         │                  │
    txt2coco.py         txt2xml.py
         │                  │
   COCO .json          VOC .xml
  （供 MMDetection     （供 VOC 框架使用）
   等框架使用）
         │
    [数据集直接使用 YOLO .txt]
         │
    GZ-DET.yaml ───── mytrain.py
         │                  │
         └──────── YOLO("yolo11.yaml")
                       │
                  model.train()
                       │
              runs/detect/train/
              ├── weights/best.pt   ← 最优模型权重
              ├── results.csv       ← 各 epoch 指标
              └── 各类可视化图表
```

---

## 五、GYU-DET 数据集质量评价
### 5.1 综合评价
| 维度 | 评分 | 说明 |
| --- | --- | --- |
| 数据规模 | ⭐⭐⭐⭐ | 11,123 张图像，在桥梁病害领域属于大规模，远超同类数据集（CODEBRIM 仅 1590 张） |
| 标注格式 | ⭐⭐⭐⭐⭐ | 使用 YOLO 标准格式，便于接入主流框架；配套提供 COCO/VOC 转换工具，兼容性强 |
| 标注精度 | ⭐⭐⭐⭐ | 三步标注流程（双人独立标注→交叉验证→专家审核），确保一致性；详细标注规范指导边界框粒度 |
| 场景多样性 | ⭐⭐⭐⭐⭐ | 覆盖白天/夜间/雨天/远景/近景等多种光照和环境；多种桥型（梁桥/拱桥/悬索桥）；多个桥梁关键部位 |
| 类别覆盖度 | ⭐⭐⭐⭐ | 6 类病害依据中国国家标准分类，涵盖最常见的桥梁表面损伤类型，与国际实践高度对应 |
| 类别平衡性 | ⭐⭐ | **显著不平衡**：Breakage（>12,000 实例）和 Reinforcement（>8,000 实例）主导数据集，而 Crack 和 Hole 实例数显著偏少 |
| 图像分辨率 | ⭐⭐⭐⭐ | 主分辨率 4608×3456（约 1600 万像素），远高于同类数据集，有利于细粒度病害的检测 |
| 公开可访问性 | ⭐⭐⭐⭐⭐ | 在 Science Data Bank 公开（CC BY-NC-ND 4.0），代码在 GitHub 开源 |


### 5.2 主要优势
**① 规模领先**  
11,123 张图像在桥梁病害检测领域属最大规模之一，相比 CODEBRIM（1,590 张）、dacl1k（1,474 张）有显著优势，能为深度学习模型提供充足的训练样本。

**② 高分辨率原图**  
主分辨率 4608×3456（尼康 D5500 拍摄），保留了丰富的细节信息，特别适合裂缝等细微病害的检测，不同于 SDNET2018 等数据集依赖滑窗裁切产生的低分辨率子图。

**③ 标注质量有保障**  
三步标注协议（独立标注→交叉校验→专家确认）+ 详细标注规范，有效减少了主观差异；允许重叠边界框，准确反映了实际工程中病害并发的复杂情形。

**④ 符合工程标准**  
病害分类依据《公路桥梁技术状况评定标准》，分类体系与工程实践直接对接；经研究者对比国际标准，病害类别与欧洲检测规范高度对应，具有国际可移植性。

**⑤ 格式标准、工具齐全**  
原生 YOLO 格式 + 官方提供的 COCO/VOC 转换工具，极大降低了使用门槛，可直接接入 YOLOv5/v8/v11、MMDetection、Detectron2 等主流框架。

### 5.3 主要局限
**① 类别严重不平衡（最大短板）**  
Breakage 实例超过 12,000，而 Hole 和 Crack 可能仅数百至千余，差距达 10 倍以上。这直接导致：

+ 论文验证实验中，Crack（mAP@0.5=0.479/0.455）和 Seepage（0.426/0.465）的检测性能明显低于 Reinforcement（0.697/0.689）
+ 模型在稀有类别上泛化能力受限，实际使用时需配合类平衡采样（Class-balanced sampling）或加权损失函数（Focal Loss）

**② 地域偏向性**  
所有图像来自贵州省（东经 103°36′–109°35′，北纬 24°37′–29°13′），该地区特殊的高原山地气候和构造特点可能使模型对其他气候带桥梁病害的泛化能力存疑。

**③ 数据采集时间较早**  
原始数据采集于 2015–2016 年，年代较久，可能缺乏近年新型桥梁材料（如 UHPC、FRP）及更复杂施工工艺带来的新型病害形态。

**④ 部分类别边界模糊**  
论文本身也承认，蜂窝面（Comb）与剥落（Breakage）在严重损伤时视觉特征相近，标注一致性存在一定挑战；轻微剥落与粗糙表面的区分也较为主观。

**⑤ txt2xml.py 类别映射存在潜在错误**  
`txt2xml.py` 中 `dic` 字典定义了 7 个类别（0~6），而 `GZ-DET.yaml` 仅定义 6 个类别（0~5），且两者的 ID 到类别名称的映射顺序不完全一致（如 `txt2xml.py` 中 `3: Comb`，而 YAML 中 `2: Comb, 3: Hole`），用户在使用 `txt2xml.py` 时需特别注意核对。

### 5.4 与同类数据集对比
| 数据集 | 图像数 | 标注类型 | 病害类别数 | 格式 |
| --- | --- | --- | --- | --- |
| GAPS | 1,969 | 图像级 | 4 | — |
| SDNET2018 | 56,092* | 图像级 | 2 | — |
| CODEBRIM | 1,590 | 边界框 | 6 | VOC |
| dacl1k | 1,474 | 边界框 | 6 | 自定义 |
| RDD2020 | 26,336 | 边界框 | 4 | VOC |
| **GYU-DET** | **11,123** | **边界框** | **6** | **YOLO** |


> * SDNET2018 的 56,092 张图像由 230 张原图裁切而来，实质内容丰富度有限。
>

GYU-DET 在病害类别覆盖（6 类，含对象检测标注）、图像分辨率（4K 级）、场景多样性等方面综合领先，是目前**最适合桥梁表面病害目标检测任务**的公开数据集之一。

### 5.5 使用建议
1. **应对类别不平衡**：训练时建议启用 `class_weights` 或使用 Focal Loss，或对稀少类别（Crack、Seepage）进行过采样。
2. **数据增强**：针对该数据集的高分辨率特点，建议使用 Mosaic、MixUp、随机裁切等增强方式，以提升模型在不同尺度下的泛化性。
3. **格式转换**：使用 `txt2xml.py` 前，务必核对 `dic` 字典中的类别 ID 映射与实际 `classes.txt` 一致，避免类别错位。
4. **评测基准**：在标准对比实验中，建议统一使用官方划分的 train/valid/test 三个子集（8:1:1），而非重新随机划分，以保证结果可复现。

---

## 六、总结
GYU-DET 代码仓库的设计思路清晰：以最简洁的训练入口（`mytrain.py`，仅 4 行有效代码）调用功能完备的 ultralytics 框架，同时提供两个独立的格式转换工具（`txt2coco.py`、`txt2xml.py`），保证数据集能跨框架使用。这种"最小业务代码 + 成熟框架"的组合是当前工程化深度学习项目的最佳实践。

数据集本身质量较高，适合用于桥梁病害检测的研究基准，但其类别不平衡问题是使用中需要重点关注的挑战。建议结合类平衡技术和数据增强手段，以充分发挥数据集的潜力。

---

_报告生成时间：2026年5月5日_  
_分析依据：论文原文（Scientific Data 2025）、代码逐行分析_

