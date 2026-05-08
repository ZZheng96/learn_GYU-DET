# Ultralytics YOLO 完整教程（从入门到实战）

# 一、基础认知：Ultralytics 与 YOLO 核心定位

## 1\.1 Ultralytics 简介

Ultralytics 是基于 PyTorch 开发的开源计算机视觉工具箱，核心支持 YOLO 全系列模型，同时集成多种视觉任务能力，提供统一的 API 接口（train/val/predict/export），无需切换框架即可完成多种 CV 任务，开箱即用、高效便捷。

核心特性：

- 底层基于 PyTorch，无需手动编写复杂训练逻辑，封装了数据集加载、优化器、损失函数等核心模块。

- 支持 YOLO 全谱系模型（YOLOv3\~YOLO11、YOLO26 等），同时兼容 SAM/FastSAM、分类、姿态检测等多种模型与任务。

- 自动生成训练日志、可视化图表，无需额外编写代码。

## 1\.2 YOLO 核心定位

YOLO（You Only Look Once）是单阶段目标检测模型，底层完全基于卷积神经网络（CNN），核心优势是「快、准、轻量」，可同时完成「目标检测（定位）\+ 分类」双任务，广泛应用于工业质检、安防、航拍、医疗等场景。

核心区别：与 Faster R\-CNN 等两阶段模型相比，YOLO 无需先生成候选框，单遍扫描图片即可同时输出目标位置、类别和置信度，砍掉冗余计算，天生具备更快的速度和更小的模型体积。

# 二、环境依赖与模型加载

## 2\.1 环境依赖（CPU/GPU 版本区分）

Ultralytics 依赖 PyTorch 运行，无需手动单独安装 PyTorch，但需根据自身硬件选择对应版本（CPU 版 / GPU 版），核心区别在于是否支持 GPU 加速训练/推理：

- 安装命令（通用）：`pip install ultralytics`（自动匹配当前环境的 PyTorch 版本）。

- 版本区分核心逻辑：
    CPU 版：无需显卡，安装后自动使用 CPU 运行，适合入门测试、小数据集调试，无需额外配置。

- GPU 版：需具备 NVIDIA 独立显卡（支持 CUDA），安装时会自动匹配 CUDA 版本的 PyTorch，训练/推理速度远快于 CPU 版，是实际项目首选。

核心逻辑：Ultralytics 是 PyTorch 的高级封装（外壳），PyTorch 是底层计算核心（内核），没有 PyTorch，Ultralytics 无法运行；GPU 版需额外确保显卡驱动与 CUDA 版本兼容（无需手动安装 CUDA，PyTorch 会自动适配）。

### 2\.1\.1 补充：GPU 版兼容性检查（避免安装失败）

1\.  检查显卡是否支持 CUDA：打开 NVIDIA 控制面板 → 帮助 → 系统信息 → 组件，查看「NVIDIA CUDA」是否有版本号（无则不支持 GPU 加速）。

2\.  若显卡支持 CUDA，安装后可通过以下代码验证 GPU 是否可用：

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
print("GPU 是否可用:", model.device)  # 输出 cuda:0 表示 GPU 可用，输出 cpu 表示仅 CPU 可用
```

## 2\.2 模型加载（YOLO 模型创建）

YOLO 模型加载有 4 种合法参数形式，核心分为「结构文件」和「权重文件」两类：

1. \.yaml 结构文件（仅定义网络结构，无权重，从零训练）

2. 示例：`model = YOLO\(\&\#34;yolo11\.yaml\&\#34;\)`

3. 作用：仅搭建模型骨架，参数随机初始化，适合从零训练（不推荐新手）。

4. 常见文件：yolo11n\.yaml、yolo11s\.yaml、yolo11m\.yaml（对应不同规模模型）。

5. \.pt 预训练权重（最常用，迁移学习）

6. 示例：`model = YOLO\(\&\#34;yolo11n\.pt\&\#34;\)`

7. 作用：加载官方在 COCO 数据集上训练好的权重，训练自己的数据集时收敛更快、效果更好（99% 场景首选）。

8. 自定义训练的 \.pt 文件（推理/继续训练）

9. 示例：`model = YOLO\(\&\#34;\./runs/detect/train/weights/best\.pt\&\#34;\)`

10. 作用：加载自己训练好的模型，用于预测、检测或继续训练。

11. 官方预设字符串（简化写法）

12. 示例：`model = YOLO\(\&\#34;yolo11n\&\#34;\)`

13. 作用：自动匹配对应的 \.pt 预训练权重，简化代码。

补充：\.pt 是 PyTorch 官方模型保存格式，包含网络结构、权重参数、类别信息，可直接加载使用。

# 三、数据集准备（标准格式与 YAML 配置）

## 3\.1 标准 YOLO 数据集格式（必遵循）

数据集需包含 train（训练集）、val（验证集）、test（测试集）三部分，每部分均包含 images（图片）和 labels（标签），目录结构如下：

```plain text
my_dataset/        # 数据集根目录
├── train/         # 训练集
│   ├── images/    # 训练图片（格式：jpg/png，如 001.jpg）
│   └── labels/    # 训练标签（格式：txt，如 001.txt）
├── val/           # 验证集
│   ├── images/    # 验证图片
│   └── labels/    # 验证标签
└── test/          # 测试集（可选，用于最终性能评估）
    ├── images/    # 测试图片
    └── labels/    # 测试标签
```

核心规则：

- 图片与标签文件名必须完全一致（如 001\.jpg 对应 001\.txt）。

- 标签格式为 YOLO 标准格式：`class\_id x y w h`（均为归一化数值，class\_id 从 0 开始编号）。

- 无需转换为 XML、JSON 格式，YOLO 可直接读取 txt 标签。

## 3\.2 YAML 配置文件（必须文件）

YAML 文件是模型与数据集的桥梁，用于告诉模型「数据在哪、有几类、类别名称」，文件名建议为 my\_data\.yaml，格式如下：

```yaml
# 1. 数据集根目录（相对路径/绝对路径均可）
path: ./my_dataset
# 2. 各数据集图片路径（自动匹配对应 labels 路径）
train: train/images  # 训练集图片路径
val: val/images      # 验证集图片路径
test: test/images    # 测试集图片路径（可选）
# 3. 类别信息
nc: 2                # 类别数量（如猫、狗两类，填 2）
names:               # 类别名称（按 class_id 顺序排列）
  0: cat
  1: dog
```

核心说明：

- 无需在 YAML 中填写 labels 路径，模型会自动根据 images 路径匹配（如 train/images → train/labels）。

- 修改时只需调整 path（数据集根目录）、nc（类别数量）、names（类别名称）三项即可。

# 四、YOLO 核心操作流程（训练→验证→测试→预测→导出）

Ultralytics 提供统一 API，所有操作仅需调用对应函数，流程固定且无需复杂配置。

## 4\.1 模型训练（train\(\)）

核心功能：用训练集训练模型，自动用验证集评分，保存最优模型和训练日志。

### 4\.1\.1 核心参数（99% 场景够用）

```python
from ultralytics import YOLO

# 1. 加载预训练模型（推荐）
model = YOLO("yolo11n.pt")

# 2. 开始训练
model.train(
    data="my_data.yaml",   # 必须：指向自己的 YAML 配置文件
    epochs=100,            # 训练轮数（默认50，根据需求调整）
    imgsz=640,             # 图片尺寸（默认640，可调整为320、480等）
    batch=8,               # 批次大小（根据显卡显存调整，显存小则调小）
    device=0,              # 训练设备（0=第一张GPU，cpu=CPU，多GPU用[0,1]）
    patience=50            # 早停机制：多少轮无精度提升则自动停止训练
)
```

### 4\.1\.2 训练硬件需求（CPU/GPU 对比）

训练硬件直接决定训练速度，不同硬件适配不同模型规模，以下为实际落地常用配置（新手参考）：

|硬件类型|推荐配置|适配模型|核心优势|适用场景|
|---|---|---|---|---|
|CPU（入门）|Intel i5/i7 10代及以上，内存 ≥16GB|yolo11n（仅小模型）|无需显卡，配置简单，零成本入门|新手调试、小数据集测试（≤1000张图）|
|GPU（首选）<br>|NVIDIA RTX 3060（8GB）及以上，内存 ≥16GB|yolo11n/s/m（主流选择）|速度快，支持中大规模模型，收敛效率高|实际项目、中等数据集（1000\-10000张图）|
|高性能GPU<br>|NVIDIA RTX 3090/4090（24GB），内存 ≥32GB|yolo11l/x（高精度需求）|支持超大模型，可设置更大batch，训练速度最快|高精度项目、大规模数据集（≥10000张图）|

补充注意：

- GPU 显存决定 batch 大小：显存越小，batch 需设越小（如 8GB 显存建议 batch=4\-8，4GB 显存建议 batch=2\-4）。

- 若 GPU 显存不足，可添加参数 `device=\&\#34;cpu\&\#34;` 切换到 CPU 训练（速度会大幅下降）。

### 4\.1\.3 训练时间预期（参考）

训练时间受 **硬件配置、模型规模、数据集大小、epochs 数量** 4个因素影响，以下为实际场景参考（统一设置 imgsz=640、epochs=100）：

|硬件|模型|数据集规模（训练集图片数）|预计训练时间|
|---|---|---|---|
|CPU（i7\-10750H）|yolo11n|500张|8\-12小时|
|CPU（i7\-10750H）|yolo11n|1000张|16\-24小时|
|GPU（RTX 3060 8GB）|yolo11n|1000张|1\-2小时|
|GPU（RTX 3060 8GB）|yolo11s|1000张|2\-3小时|
|GPU（RTX 3060 8GB）|yolo11m|1000张|3\-5小时|
|GPU（RTX 4090 24GB）|yolo11x|10000张|8\-12小时|

补充说明：

- epochs 越多，训练时间越长（如 epochs=200，时间约为 epochs=100 的2倍）。

- 图片尺寸越大（如 imgsz=1280），训练时间约增加1\.5\-2倍。

- 早停机制（patience）会提前终止训练（若精度不再提升），实际时间可能短于预期。

### 4\.1\.4 train\(\) 函数详细参数（全场景覆盖）

train\(\) 函数参数繁多，以下按「必选参数、核心可选参数、进阶参数」分类说明，新手优先掌握必选\+核心可选参数即可：

#### （1）必选参数（缺一不可）

- `data`：str 类型，必填。指向自己的 YAML 配置文件路径（相对路径/绝对路径均可），用于告诉模型数据集位置、类别信息。
    
示例：`data=\&\#34;my\_data\.yaml\&\#34;`、`data=\&\#34;D:/datasets/my\_data\.yaml\&\#34;`

#### （2）核心可选参数（高频使用）

- `epochs`：int 类型，默认50。训练轮数，即模型完整遍历数据集的次数。

说明：新手建议50\-100轮，数据集规模大（≥10000张）可设100\-200轮，配合早停机制避免过拟合。


- `imgsz`：int 类型，默认640。输入图片的尺寸（正方形），需为32的整数倍（如320、480、640、1280）。
    
说明：尺寸越大，精度可能越高，但训练速度越慢、显存占用越多；新手默认640即可，小目标较多可设800\-1280。
  

- `batch`：int 类型，默认16。每一批次训练的图片数量，核心受 GPU 显存限制。
    
说明：8GB 显存建议4\-8，4GB 显存建议2\-4，CPU 建议2\-4；显存不足时可设 `batch=\-1`，自动匹配最大可用 batch。
  

- `device`：str/int/list 类型，默认\&\#34;auto\&\#34;（自动检测可用设备）。训练设备指定。
    
示例：`device=0`（使用第一张 GPU）、`device=\&\#34;cpu\&\#34;`（使用 CPU）、`device=\[0,1\]`（使用两张 GPU 并行训练）。
  

- `patience`：int 类型，默认10。早停机制参数，指连续多少轮验证集精度（mAP）无提升，就自动终止训练，避免过拟合。
    
说明：新手建议30\-50，数据集波动大时可设更大（如100）。
  

- `weight`：str 类型，默认None。预训练权重路径，若不指定，将从零训练（不推荐新手）。

示例：`weight=\&\#34;yolo11n\.pt\&\#34;`、`weight=\&\#34;\./best\.pt\&\#34;`。
  

- `name`：str 类型，默认\&\#34;train\&\#34;。训练任务名称，用于命名训练日志、权重文件的保存目录（runs/detect/name/）。
    
示例：`name=\&\#34;cat\_dog\_train\&\#34;`，方便区分不同训练任务。
  

- `save`：bool 类型，默认True。是否保存训练过程中的权重文件、日志和可视化图表。
    
说明：建议保持True，便于后续查看训练效果、复用模型。
  

#### （3）进阶参数（按需使用）

- `lr0`：float 类型，默认0\.01。初始学习率，控制参数更新速度，过大易不收敛，过小训练过慢。
    
说明：新手默认即可，如需调整，建议在0\.001\-0\.05之间微调。
  

- `lrf`：float 类型，默认0\.01。最终学习率（lr0 \* lrf），用于学习率衰减。
  

- `momentum`：float 类型，默认0\.937。动量参数，加速梯度下降，默认值适配大多数场景。
  

- `weight\_decay`：float 类型，默认0\.0005。权重衰减，用于防止过拟合，数值越大，正则化越强。
  

- `warmup\_epochs`：int 类型，默认3。热身轮数，训练初期用小学习率逐步适应，避免模型不收敛。
  

- `augment`：bool 类型，默认True。是否开启数据增强（随机裁剪、翻转、缩放等），提升模型泛化能力。
    
说明：数据集较小时建议开启，数据集较大时可根据需求关闭。

- `dropout`：float 类型，默认0\.0。 dropout 概率，用于防止过拟合，数值越大， dropout 越强（建议0\.0\-0\.2之间）。
  

- `workers`：int 类型，默认8。数据加载线程数，数值越大，数据加载越快，受 CPU 核心数限制。
    
说明：CPU 核心数少（≤4）建议设2\-4，避免卡顿。
  

### 4\.1\.5 训练输出物（自动生成）

训练完成后，自动在`runs/detect/train/` 目录下生成以下文件：

- 权重文件：best\.pt（所有轮次中验证集分数最高的模型，首选使用）、last\.pt（最后一轮训练的模型，用于断点续训）。

- 日志与可视化：results\.csv（每一轮的训练/验证分数）、results\.png（loss、mAP 曲线）、PR\_curve\.png（精度\-召回曲线）、confusion\_matrix\.png（混淆矩阵）。

- 预测效果图：val\_batch0\.jpg（验证集预测示例图）。

### 4\.1\.6 断点续训

若训练中断（断电、死机），可加载 last\.pt 继续训练，无需从头开始：

```python
model = YOLO("last.pt")
model.train(resume=True)  # resume=True 表示继续训练
# 可选：继续训练时可调整参数（如epochs、batch）
model.train(resume=True, epochs=150, batch=6)
```

## 4\.2 模型验证（val\(\)）

核心功能：评估模型精度，输出量化指标，支持训练集、验证集、测试集的评分。

### 4\.2\.1 核心用法

```python
import os
from ultralytics import YOLO

# 1. 加载最优模型
model = YOLO("best.pt")

# 2. 定义代码文件夹路径（根据自身项目结构调整，确保路径正确）
code_folder = os.path.dirname(os.path.abspath(__file__))  # 自动获取当前代码文件所在文件夹

# 3. 完整版验证代码（含你需要的所有参数，可直接复制运行）
metrics = model.val(
    data="my_data.yaml",       # 必须：指向自己的YAML配置文件
    split="test",              # 关键：用测试集进行最终性能评估
    save=True,                 # 保存画框图（直观查看检测效果）
    save_txt=True,             # 保存检测结果（txt格式，用于后续分析）
    imgsz=640,                 # 输入图片转为640*640（与训练时尺寸一致）
    device="cpu",              # 验证设备（有GPU可改为device=0）
    conf=0.25,                 # 置信度阈值（过滤低置信度误检框）
    iou=0.6,                   # IOU阈值（NMS去重，减少重复检测框）
    project=os.path.join(code_folder,"../results"),    # 输出根目录（自定义路径）
    name="test_result",   # 输出文件夹名称（方便管理验证结果）
    plots=True                 # 生成PR曲线、混淆矩阵等可视化图表
)

# 4. 输出核心评分指标（量化模型性能）
print("测试集 mAP50:", metrics.box.map50)      # 最关键指标，综合精度
print("测试集 mAP50-95:", metrics.box.map)     # 更严格的综合精度
print("测试集 精确率 Precision:", metrics.box.p)# 预测框中正确的比例（误检率）
print("测试集 召回率 Recall:", metrics.box.r)  # 真实目标中被检测到的比例（漏检率）
```

### 4\.2\.2 关键参数：split

split 用于指定「评分的数据集分片」，对应 YAML 文件中的 train/val/test 路径：

- split=\&\#34;train\&\#34;：用训练集评分（较少用）。

- split=\&\#34;val\&\#34;：用验证集评分（默认，训练中自动调用）。

- split=\&\#34;test\&\#34;：用测试集评分（最终性能评估，项目/论文首选）。

### 4\.2\.3 val\(\) 函数详细参数

val\(\) 函数参数主要用于控制验证过程的精度、速度，以下为核心参数说明：

- `data`：str 类型，必填。指向 YAML 配置文件路径，与 train\(\) 函数的 data 参数一致。

- `split`：str 类型，默认\&\#34;val\&\#34;。指定验证的数据集分片（train/val/test）。

- `imgsz`：int 类型，默认640。验证时的图片尺寸，需与训练时的 imgsz 一致，否则会影响精度评估。

- `batch`：int 类型，默认16。验证时的批次大小，受 GPU 显存限制，默认即可，显存不足可调小。

- `device`：str/int 类型，默认\&\#34;auto\&\#34;。验证设备，与 train\(\) 函数的 device 参数一致，建议与训练设备相同。

- `conf`：float 类型，默认0\.001。置信度阈值，只有预测置信度高于该值的框才会参与评分，新手默认即可。

- `iou`：float 类型，默认0\.6。IOU 阈值，用于 NMS 去重和精度计算，默认0\.6适配大多数场景。

- `save\_json`：bool 类型，默认False。是否保存验证结果为 JSON 文件，用于后续论文/项目汇报。

- `save\_conf`：bool 类型，默认False。是否在预测框上显示置信度数值，用于可视化验证效果。

- `plots`：bool 类型，默认True。是否生成验证可视化图表（PR曲线、混淆矩阵等），建议保持True。

## 4\.3 模型预测（predict\(\)）

核心功能：用训练好的模型进行推理，生成带检测框的图片/视频，不输出量化评分（仅用于可视化效果）。

### 核心用法

```python
model = YOLO("best.pt")

# 1. 单张图片预测（保存带框图片）
model.predict("test.jpg", save=True)

# 2. 视频预测（保存带框视频）
model.predict("test.mp4", save=True)

# 3. 文件夹批量预测
model.predict("./test_images", save=True)
```

输出路径：预测结果自动保存到 `runs/detect/predict/` 目录下，包含带检测框的图片/视频。

### 补充：预测速度参考（CPU/GPU 对比）

- CPU（i7\-10750H）：单张 640×640 图片，预测时间约 0\.5\-1 秒/张。

- GPU（RTX 3060）：单张 640×640 图片，预测时间约 0\.01\-0\.05 秒/张（实时推理）。

- 视频预测：GPU 可支持 30\+ 帧/秒（流畅），CPU 仅支持 2\-5 帧/秒（卡顿）。

### 4\.3\.1 predict\(\) 函数详细参数

predict\(\) 函数参数主要用于控制预测效果、输出格式，以下为高频参数说明：

#### （1）必选相关参数（输入源）

- `source`：str 类型，必填。预测输入源，可传入单张图片路径、视频路径、文件夹路径、摄像头索引（0为默认摄像头）。
    
示例：`source=\&\#34;test\.jpg\&\#34;`、`source=\&\#34;test\.mp4\&\#34;`、`source=\&\#34;\./test\_imgs\&\#34;`、`source=0`（摄像头实时预测）。
  

#### （2）核心可选参数（控制预测效果）

- `imgsz`：int 类型，默认640。预测时的图片尺寸，需与训练时一致，否则会影响预测精度。

- `conf`：float 类型，默认0\.25。置信度阈值，只有预测置信度高于该值的框才会显示，可调整以减少误检/漏检。
    
说明：误检多则提高 conf（如0\.5），漏检多则降低 conf（如0\.1）。
  

- `iou`：float 类型，默认0\.7。IOU 阈值，用于 NMS 去重，数值越小，去重越严格（减少重复框）。

- `device`：str/int 类型，默认\&\#34;auto\&\#34;。预测设备，GPU 预测速度远快于 CPU，实时预测建议用 GPU。

- `classes`：list 类型，默认None。指定只预测某几类目标，用于过滤无关目标。
    
示例：`classes=\[0\]`（只预测类别ID为0的目标，对应 YAML 中的第一个类别）。


#### （3）输出控制参数

- `save`：bool 类型，默认False。是否保存预测结果（带框图片/视频），需要保存则设为True。

- `save\_dir`：str 类型，默认\&\#34;runs/detect/predict\&\#34;。预测结果保存目录，可自定义路径。
    
示例：`save\_dir=\&\#34;\./predict\_results\&\#34;`。
  

- `show`：bool 类型，默认False。是否实时显示预测结果（图片/视频），适合调试查看效果。

- `save\_conf`：bool 类型，默认False。是否在预测框上显示置信度数值。

- `save\_txt`：bool 类型，默认False。是否保存预测结果为 txt 标签文件（YOLO 格式），用于后续分析。

- `line\_width`：int 类型，默认3。预测框的线条宽度，数值越大，框越粗，便于查看。

## 4\.4 模型导出（export\(\)）

核心功能：将 \.pt 模型导出为部署格式，用于实际项目落地（如嵌入式、手机端）。

### 核心用法

```python
model = YOLO("best.pt")
# 导出为 ONNX 格式（最常用，跨平台部署）
model.export(format="onnx")
# 其他支持格式：TensorRT、CoreML、TensorFlow、OpenVINO 等
# model.export(format="engine")  # TensorRT 格式（速度最快，仅支持 GPU 部署）
```

### 4\.4\.1 export\(\) 函数详细参数

export\(\) 函数参数主要用于控制导出格式、精度、速度，以下为核心参数说明：

- `format`：str 类型，必填。导出格式，支持 onnx、engine（TensorRT）、coreml、tf（TensorFlow）、openvino 等。
    
常用格式：`format=\&\#34;onnx\&\#34;`（跨平台通用）、`format=\&\#34;engine\&\#34;`（GPU 部署，速度最快）。
  

- `imgsz`：int 类型，默认640。导出模型的输入尺寸，需与训练时一致。

- `batch`：int 类型，默认1。导出模型的批次大小，1为通用批次，可根据部署需求调整。

- `device`：str/int 类型，默认\&\#34;auto\&\#34;。导出设备，导出 TensorRT 格式需用 GPU（device=0）。

- `half`：bool 类型，默认False。是否导出半精度模型（FP16），体积更小、速度更快，GPU 部署建议开启。

- `int8`：bool 类型，默认False。是否导出 INT8 量化模型，体积最小、速度最快，但精度会有少量损失，嵌入式部署首选。

- `simplify`：bool 类型，默认True。是否简化模型结构，减小模型体积，提升部署速度，建议保持True。

- `export\_dir`：str 类型，默认\&\#34;runs/detect/export\&\#34;。模型导出目录，可自定义路径。

# 五、YOLO 底层原理

## 5\.1 核心架构：卷积神经网络（CNN）

YOLO 底层完全基于 CNN 构建，从输入到输出全是 CNN 组件，无其他架构：

- 核心组件：普通卷积（Conv）、深度可分离卷积、残差卷积（Residual）、瓶颈模块（Bottleneck）、SPPF 特征融合、PANet 多尺度拼接。

- 特征提取逻辑：低层卷积提取边缘、纹理等基础特征；中层卷积提取形状、局部结构特征；高层卷积提取物体整体语义特征。

## 5\.2 检测\+分类双任务原理

YOLO 是「多任务卷积神经网络」，单遍前向传播即可同时完成两个任务，核心流程：

1. 将输入图片划分为多个均匀网格。

2. 每个网格同时预测：① 目标类别（分类）；② 目标边框坐标（x,y,w,h，定位）；③ 置信度（判断网格内是否有目标）。

3. 通过 NMS（非极大值抑制）去除重复检测框，输出最终结果。

核心优势：无需分阶段处理，一步完成检测\+分类，效率远高于两阶段模型。

## 5\.3 YOLO 快、小的核心原因

1. 单阶段架构：无需生成候选框，单遍扫描即可输出结果，砍掉冗余计算。

2. 轻量化设计：采用精简卷积、深度可分离卷积等结构，减少模型参数和计算量。

3. 高效特征融合：通过 SPPF、PANet 融合多尺度特征，无需重复扫描图片，兼顾速度和精度。

# 六、进阶拓展

## 6\.1 YOLO 模型选型（速度与精度平衡）

YOLO 各版本按「参数量/速度/精度」分为 n（nano）、s（small）、m（medium）、l（large）、x（extra large）五个档位，选型口诀：

- 边缘部署、追求速度 → yolo11n（最小最快，牺牲少量精度）。

- 日常场景、平衡速度精度 → yolo11s / yolo11m（首选）。

- 追求最高精度、不在乎速度 → yolo11x（拉满精度，媲美两阶段模型）。

## 6\.2 YOLO 精度相关说明

### 6\.2\.1 速度与精度的取舍

- 轻量版（n/s）：为速度牺牲少量精度，主要影响小目标、遮挡目标的检测效果。

- 中大版（m/l/x）：几乎不牺牲精度，精度可追平甚至超越传统两阶段模型（如 Faster R\-CNN）。

### 6\.2\.2 精度上限与 95% 正确率

- 简单场景（背景干净、目标清晰、遮挡少、单一品类）：YOLO 可轻松达到 Precision 95%\+、mAP50 95%\+（如工业零件缺陷检测、固定场景人车检测）。

- 复杂场景（航拍小目标、密集人群、严重遮挡、暗光、雨雪天气）：任何模型都有上限，YOLO 最高可达到 mAP50 70%\~85%，无法达到 95%。

- YOLO 精度上限：同等参数量下，YOLO 是单阶段模型中的天花板，放大到超大版本（yolo11x），精度可媲美高端两阶段模型。

## 6\.3 Ultralytics 其他支持的模型与任务

Ultralytics 不止支持 YOLO，还集成多种 CV 模型与任务，统一 API 调用：

1. 分割模型：SAM（分割一切）、FastSAM（Ultralytics 优化版，快且准），用于高精度抠图、标注。
    `from ultralytics import FastSAM
model = FastSAM\(\&\#34;FastSAM\-s\.pt\&\#34;\)
``model\.predict\(\&\#34;img\.jpg\&\#34;, save=True\)`

2. 分类模型：支持 ResNet、EfficientNet、MobileNet 等，用于图像分类、缺陷识别（仅分类，不画框）。`model = YOLO\(\&\#34;yolov8n\-cls\.pt\&\#34;\)
``model\.train\(data=\&\#34;my\_cls\_data\&\#34;, epochs=10\)`

3. 姿态关键点检测：支持 YOLO\-Pose、HRNet，用于人体 2D/3D 关键点检测（如动作识别、健身分析）。
    `model = YOLO\(\&\#34;yolov8n\-pose\.pt\&\#34;\)
``model\.predict\(\&\#34;person\.jpg\&\#34;, save=True\)`

4. 旋转框检测（OBB）：支持 YOLO\-OBB，用于航拍、文本、车牌等倾斜目标检测。

5. 多目标跟踪：内置 ByteTrack、DeepSORT，用于视频流中目标跟踪（如人流、车流统计）。
    `model\.track\(\&\#34;video\.mp4\&\#34;, save=True, tracker=\&\#34;bytetrack\.yaml\&\#34;\)`

6. 特征提取：用 YOLO/SAM 做图像特征提取，用于以图搜图、相似度比对。

## 6\.4 常见误区澄清

- 误区1：YOLO 是边缘检测模型 → 错误。YOLO 是语义检测模型，底层用卷积自动学习边缘特征，而非传统边缘检测（Canny/Sobel）。

- 误区2：Ultralytics 可以不用 PyTorch → 错误。Ultralytics 底层基于 PyTorch，安装时会自动安装 PyTorch，无 PyTorch 无法运行。

- 误区3：predict\(\) 可以用于测试集评分 → 错误。predict\(\) 仅用于可视化，测试集评分必须用 val\(split=\&\#34;test\&\#34;\)。

- 误区4：“riss” 是 YOLO 相关模型 → 错误。“riss” 是 RS/IRST 的口误，RS\-YOLO 是遥感专用 YOLO，IRST\-YOLO 是红外小目标检测专用 YOLO。

- 误区5：Edge YOLO 是边缘检测 → 错误。Edge YOLO 是为边缘设备（手机、嵌入式）优化的轻量 YOLO，强调速度和低功耗。

- 误区6：GPU 训练一定比 CPU 好 → 不完全对。GPU 优势在于速度，若仅调试代码、测试小模型，CPU 配置简单、零成本，无需特意配置 GPU。

- 误区7：训练时间越长，模型精度越高 → 错误。训练到一定轮次后，模型会过拟合（训练集精度高，验证集精度下降），早停机制会自动终止训练，无需盲目增加 epochs。

## 6\.5 用 PyTorch 改进 YOLO 训练效果（核心实战技巧）

Ultralytics YOLO 底层基于 PyTorch，可通过 PyTorch 的原生功能（优化器、损失函数、正则化、数据增强等）自定义改进，大幅提升模型训练效果，以下为可直接落地的实战方法：

### 6\.5\.1 自定义 PyTorch 优化器（替换默认优化器）

YOLO 默认使用 SGD 优化器，可替换为 PyTorch 中的 Adam、AdamW 等优化器，适配不同数据集，加速收敛、提升精度（尤其适合小数据集）。

```python
from ultralytics import YOLO
import torch.optim as optim

# 1. 加载模型
model = YOLO("yolo11n.pt")

# 2. 自定义 PyTorch 优化器（AdamW，比默认 SGD 更易收敛）
optimizer = optim.AdamW(
    model.parameters(),  # 模型参数
    lr=0.001,            # 学习率（可微调）
    weight_decay=0.0005  # 权重衰减，防止过拟合
)

# 3. 训练时指定自定义优化器
model.train(
    data="my_data.yaml",
    epochs=100,
    batch=8,
    optimizer=optimizer,  # 传入自定义优化器
    lr0=0.001,           # 需与优化器的 lr 一致
    lrf=0.01
)
```

补充说明：

- 小数据集/难收敛场景：优先用 AdamW（收敛快，不易震荡）。

- 大数据集/稳定场景：可用 SGD（默认）\+ 动量（momentum=0\.937），泛化能力更强。

- 可结合 PyTorch 的 `optim\.lr\_scheduler` 自定义学习率调度策略（如余弦退火、StepLR），进一步优化收敛。

### 6\.5\.2 自定义 PyTorch 损失函数（解决类别不平衡）

YOLO 默认损失函数为 CIoU Loss，当数据集存在类别不平衡（某类目标极少）、小目标过多时，可通过 PyTorch 自定义损失函数，提升小众类别、小目标的检测精度。

```python
from ultralytics import YOLO
import torch
import torch.nn as nn

# 1. 自定义损失函数（结合 Focal Loss 解决类别不平衡）
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bbox_loss = nn.CIoULoss()  # 边框损失（默认）
        self.cls_loss = nn.FocalLoss(alpha=0.25, gamma=2)  # 分类损失（Focal Loss）
        self.conf_loss = nn.BCEWithLogitsLoss()  # 置信度损失

    def forward(self, preds, targets):
        # preds：模型预测输出；targets：真实标签
        bbox_pred, cls_pred, conf_pred = preds[0], preds[1], preds[2]
        bbox_target, cls_target, conf_target = targets[0], targets[1], targets[2]
        
        # 计算各部分损失，加权求和（可调整权重）
        bbox_loss = self.bbox_loss(bbox_pred, bbox_target)
        cls_loss = self.cls_loss(cls_pred, cls_target.long())
        conf_loss = self.conf_loss(conf_pred, conf_target)
        
        total_loss = 1.0 * bbox_loss + 1.0 * cls_loss + 0.5 * conf_loss
        return total_loss

# 2. 加载模型并替换损失函数
model = YOLO("yolo11n.pt")
model.loss = CustomLoss()  # 替换为自定义损失函数

# 3. 正常训练
model.train(
    data="my_data.yaml",
    epochs=100,
    batch=8
)
```

核心优势：Focal Loss 可降低易分类样本的权重，提升难分类样本（小众类别、小目标）的损失占比，解决类别不平衡导致的漏检问题。

### 6\.5\.3 利用 PyTorch 自定义数据增强（提升模型泛化能力）

Ultralytics YOLO 虽默认开启基础数据增强，但针对特定场景（如光照变化、目标角度偏移、遮挡等），可通过 PyTorch 原生数据增强工具（torchvision\.transforms）自定义增强策略，进一步提升模型的抗干扰能力和泛化能力，尤其适合小数据集、场景复杂的任务。

核心思路：通过自定义数据增强管道，对训练集图片进行随机变换（如随机调整亮度、旋转、裁剪、翻转等），让模型学习到更多场景下的目标特征，减少过拟合，提升实际部署时的鲁棒性。

```python
from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image
import os

# 1. 自定义数据增强管道（适配YOLO训练，可根据需求调整）
custom_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=640, scale=(0.8, 1.2)),  # 随机裁剪+缩放，保持图片尺寸640
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转，概率50%
    transforms.RandomVerticalFlip(p=0.2),   # 随机垂直翻转，概率20%
    transforms.RandomRotation(degrees=15),  # 随机旋转±15度，避免目标角度偏移影响检测
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 随机调整亮度、对比度、饱和度
    transforms.ToTensor(),  # 转换为Tensor格式（PyTorch训练必需）
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图像归一化
])

# 2. 自定义数据集加载器（集成自定义数据增强）
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # 加载图片
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = Image.open(img_path).convert('RGB')
        # 加载对应标签（YOLO格式txt文件）
        label_path = os.path.join(self.label_dir, self.img_names[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(label_path, 'r') as f:
            labels = [list(map(float, line.strip().split())) for line in f.readlines()]
        labels = torch.tensor(labels, dtype=torch.float32)  # 转换为Tensor格式

        # 应用自定义数据增强（仅对图片进行，标签无需增强）
        if self.transform:
            img = self.transform(img)

        return img, labels

# 3. 加载模型并替换数据集加载器
model = YOLO("yolo11n.pt")

# 4. 配置自定义数据集路径
train_img_dir = "./my_dataset/train/images"
train_label_dir = "./my_dataset/train/labels"
val_img_dir = "./my_dataset/val/images"
val_label_dir = "./my_dataset/val/labels"

# 5. 创建自定义数据集实例
train_dataset = CustomDataset(train_img_dir, train_label_dir, transform=custom_transform)
val_dataset = CustomDataset(val_img_dir, val_label_dir, transform=transforms.ToTensor())  # 验证集不增强，保证评估准确

# 6. 创建数据加载器（批量加载数据，配合训练）
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,  # 训练集打乱，提升训练效果
    num_workers=4
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,  # 验证集不打乱
    num_workers=4
)

# 7. 自定义训练循环（结合PyTorch原生训练逻辑，灵活控制训练过程）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
criterion = model.loss  # 可使用自定义损失函数或默认损失函数
epochs = 100

for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # 前向传播
        preds = model(imgs)
        # 计算损失
        loss = criterion(preds, labels)
        # 反向传播+参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * imgs.size(0)
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            preds = model(imgs)
            loss = criterion(preds, labels)
            val_loss += loss.item() * imgs.size(0)
    
    # 打印每轮训练/验证损失
    train_loss_avg = train_loss / len(train_dataset)
    val_loss_avg = val_loss / len(val_dataset)
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}")
    
    # 保存最优模型（根据验证损失）
    if (epoch == 0) or (val_loss_avg < best_val_loss):
        best_val_loss = val_loss_avg
        model.save("./best_custom_aug.pt")
        print("Saved best model with custom data augmentation!")

```

核心说明：

- 数据增强仅作用于训练集，验证集和测试集不进行增强，确保评估结果的准确性，避免因增强导致的精度误判。

- 可根据数据集特点调整增强策略：如航拍数据集可增加旋转角度、缩放范围；暗光数据集可重点调整亮度、对比度；遮挡场景可增加随机遮挡增强（需额外添加 transforms\.RandomErasing 等操作）。

- 自定义训练循环的优势的是：可灵活结合自定义优化器、损失函数、数据增强，精准适配自己的数据集，比默认 train\(\) 函数更具针对性，尤其适合复杂场景的模型训练。

补充技巧：若不想自定义完整训练循环，也可通过 Ultralytics 的 `augment` 参数结合自定义增强函数，简化代码，兼顾灵活性和便捷性。

> （注：文档部分内容可能由 AI 生成）
