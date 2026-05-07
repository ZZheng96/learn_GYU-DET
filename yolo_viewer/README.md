# YOLO标注查看器

一个用于查看YOLO格式标注图片的桌面应用程序，基于Python + tkinter开发。

## 功能特性

- **图片加载**: 选择图片文件夹和对应的YOLO标签文件夹
- **YAML配置支持**: 从YAML配置文件自动读取类别和数据集路径（可选，需安装PyYAML）
- **标注显示**: 自动在图片上绘制YOLO格式的边界框
- **文件列表**: 左侧显示所有图片，双击可快速跳转
- **导航功能**:
  - 首张/末张/上一张/下一张按钮
  - 键盘快捷键：左右箭头、上下箭头
- **搜索功能**: 按图片名称搜索并跳转
- **类别管理**: 可自定义编辑类别名称、显示类别列表
- **全屏显示**: 支持全屏模式查看
- **自适应缩放**: 图片自动适配画布大小
- **自适应窗口**: 初始窗口1400x900，最小尺寸1200x800

## 安装依赖

**基础依赖**（必需）：
```bash
pip install Pillow
```

**可选依赖**（支持YAML配置文件）：
```bash
pip install PyYAML
```

或直接安装所有依赖：
```bash
pip install Pillow PyYAML
```

## 运行程序
```bash
python yolo_viewer.py
```

### 注意事项

- 确保 Python 已正确安装并添加到系统 PATH
- 运行前请先安装依赖：`pip install -r requirements.txt`

## 使用说明

### 方式一：使用YAML配置文件（需安装PyYAML）

1. 安装 PyYAML：`pip install PyYAML`
2. 点击"YAML文件"右侧的"浏览"按钮
3. 选择YOLO数据集的配置文件（如 `data.yaml`）
4. 程序会自动：
   - 加载类别列表
   - 自动填充图片和标签路径
5. 点击"加载数据"加载图片

### 方式二：手动选择路径

1. 点击"图片路径"右侧的"浏览"选择图片文件夹
2. 点击"标签路径"右侧的"浏览"选择标签文件夹（可选）
3. 点击"加载数据"按钮

### YAML文件格式支持

支持标准的YOLO数据集YAML格式：

```yaml
# 方式1: 字典格式
path: /path/to/dataset
train: images/train
names:
  0: person
  1: car
  2: bicycle

# 方式2: 列表格式
path: /path/to/dataset
train: images/train
names: ['person', 'car', 'bicycle']
```

### 标注文件格式

- 标签文件为.txt格式
- 文件名需与图片文件名一致（不包括扩展名）
- 例如：`image001.jpg` 对应 `image001.txt`

### YOLO标注格式

每行一个标注，格式为：
```
<class_id> <x_center> <y_center> <width> <height>
```
坐标值为归一化值（0-1）

### 键盘快捷键

| 快捷键 | 功能 |
|--------|------|
| `←` `→` | 上一张/下一张 |
| `↑` `↓` | 首张/末张 |
| `F11` | 切换全屏 |
| `Esc` | 退出全屏 |

## 目录结构

```
yolo_viewer/
├── yolo_viewer.py      # 主程序
├── requirements.txt    # 依赖列表
└── README.md          # 说明文档
```

## 示例数据结构

```
dataset/
├── data.yaml          # 配置文件
├── train/
│   ├── images/
│   │   ├── cat001.jpg
│   │   ├── cat002.jpg
│   │   └── dog001.jpg
│   └── labels/
|       ├── cat001.txt    # 内容: 0 0.5 0.5 0.8 0.8
│       ├── cat002.txt
│       └── dog001.txt    # 内容: 1 0.3 0.4 0.6 0.7
└── val/
    ├── images/
    └── labels/
```
