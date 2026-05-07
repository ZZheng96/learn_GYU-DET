"""
YOLO标注查看器
支持查看YOLO格式标注的图片，具有完整的UI界面
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import glob


class YoloViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO标注查看器")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        # 检查依赖
        self.yaml_available = self._check_yaml()

        # 数据
        self.image_dir = None
        self.label_dir = None
        self.yaml_path = None
        self.image_files = []
        self.label_files = {}
        self.current_index = 0
        self.class_names = []
        self.class_colors = {}

        # 类别配置（可扩展）
        self.default_classes = [
            "person", "car", "bicycle", "motorcycle", "bus", "truck",
            "traffic_light", "stop_sign", "cat", "dog", "bird"
        ]

        self.setup_ui()
        self.load_default_config()

    def _check_yaml(self):
        """检查PyYAML是否可用"""
        try:
            import yaml
            return True
        except ImportError:
            return False

    def setup_ui(self):
        # 顶部工具栏
        toolbar = ttk.Frame(self.root, padding="5")
        toolbar.pack(fill=tk.X)

        # 第一行：路径选择
        row1 = ttk.Frame(toolbar)
        row1.pack(fill=tk.X, pady=2)

        # 图片路径选择
        ttk.Label(row1, text="图片路径:", width=10).pack(side=tk.LEFT, padx=2)
        self.image_path_var = tk.StringVar()
        image_path_entry = ttk.Entry(row1, textvariable=self.image_path_var, width=50)
        image_path_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="浏览", command=self.select_image_dir, width=6).pack(side=tk.LEFT, padx=2)

        # 标签路径选择
        ttk.Label(row1, text="标签路径:", width=10).pack(side=tk.LEFT, padx=2)
        self.label_path_var = tk.StringVar()
        label_path_entry = ttk.Entry(row1, textvariable=self.label_path_var, width=50)
        label_path_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="浏览", command=self.select_label_dir, width=6).pack(side=tk.LEFT, padx=2)

        # 第二行：YAML和加载
        row2 = ttk.Frame(toolbar)
        row2.pack(fill=tk.X, pady=2)

        # YAML文件选择
        ttk.Label(row2, text="YAML文件:", width=10).pack(side=tk.LEFT, padx=2)
        self.yaml_path_var = tk.StringVar()
        yaml_path_entry = ttk.Entry(row2, textvariable=self.yaml_path_var, width=50)
        yaml_path_entry.pack(side=tk.LEFT, padx=2)

        if self.yaml_available:
            ttk.Button(row2, text="浏览", command=self.select_yaml_file, width=6).pack(side=tk.LEFT, padx=2)
        else:
            yaml_path_entry.config(state=tk.DISABLED)
            ttk.Label(row2, text="(未安装PyYAML)", foreground="red").pack(side=tk.LEFT, padx=2)

        # 加载按钮
        ttk.Button(row2, text="加载数据", command=self.load_data, width=10).pack(side=tk.LEFT, padx=15)
        ttk.Separator(row2, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # 类别编辑
        ttk.Button(row2, text="编辑类别", command=self.edit_classes, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="显示类别", command=self.show_classes, width=10).pack(side=tk.LEFT, padx=2)

        # 搜索栏
        search_frame = ttk.Frame(self.root, padding="5")
        search_frame.pack(fill=tk.X)
        ttk.Label(search_frame, text="搜索图片名:", width=12).pack(side=tk.LEFT, padx=2)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=40)
        search_entry.pack(side=tk.LEFT, padx=2)
        search_entry.bind("<Return>", lambda e: self.search_image())
        ttk.Button(search_frame, text="搜索", command=self.search_image, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(search_frame, text="清除", command=self.clear_search, width=8).pack(side=tk.LEFT, padx=2)

        # 状态栏
        status_frame = ttk.Frame(self.root, padding="0 5 5 5")
        status_frame.pack(fill=tk.X)
        self.status_var = tk.StringVar()
        status_text = "请选择图片和标签路径"
        if not self.yaml_available:
            status_text += " (提示: 安装PyYAML可使用YAML配置功能)"
        self.status_var.set(status_text)
        ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X)

        # 中间区域：图片显示 + 文件列表
        content = ttk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧：文件列表
        left_panel = ttk.Frame(content, width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        ttk.Label(left_panel, text="文件列表").pack(anchor=tk.W)
        self.file_listbox = tk.Listbox(left_panel, width=35)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(left_panel, orient=tk.VERTICAL, command=self.file_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=scrollbar.set)
        self.file_listbox.bind("<Double-Button-1>", lambda e: self.show_selected_file())

        # 右侧：图片显示区域
        right_panel = ttk.Frame(content)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(right_panel, bg="gray20")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 底部控制栏
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X, side=tk.BOTTOM)

        ttk.Button(control_frame, text="<< 首张", command=self.first_image, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="< 上一张", command=self.prev_image, width=10).pack(side=tk.LEFT, padx=5)

        self.page_info_var = tk.StringVar()
        self.page_info_var.set("0 / 0")
        ttk.Label(control_frame, textvariable=self.page_info_var, width=15).pack(side=tk.LEFT, padx=10)

        ttk.Button(control_frame, text="下一张 >", command=self.next_image, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="末张 >>", command=self.last_image, width=10).pack(side=tk.LEFT, padx=5)

        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=20)

        ttk.Button(control_frame, text="刷新", command=self.refresh_current, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="全屏", command=self.toggle_fullscreen, width=10).pack(side=tk.LEFT, padx=5)

        # 绑定键盘事件
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<Up>", lambda e: self.first_image())
        self.root.bind("<Down>", lambda e: self.last_image())
        self.root.bind("<Escape>", lambda e: self.exit_fullscreen())
        self.root.bind("<F11>", lambda e: self.toggle_fullscreen())

        self.is_fullscreen = False
        self.current_image_tk = None
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def load_default_config(self):
        # 生成默认颜色
        import random
        for cls in self.default_classes:
            self.class_names.append(cls)
            self.class_colors[cls] = "#{:06x}".format(random.randint(0, 0xFFFFFF))

    def select_image_dir(self):
        dir_path = filedialog.askdirectory(title="选择图片文件夹")
        if dir_path:
            self.image_path_var.set(dir_path)

    def select_label_dir(self):
        dir_path = filedialog.askdirectory(title="选择标签文件夹")
        if dir_path:
            self.label_path_var.set(dir_path)

    def select_yaml_file(self):
        if not self.yaml_available:
            messagebox.showerror("错误", "PyYAML未安装，请运行: pip install pyyaml")
            return

        file_path = filedialog.askopenfilename(
            title="选择YAML配置文件",
            filetypes=[("YAML文件", "*.yaml *.yml"), ("所有文件", "*.*")]
        )
        if file_path:
            self.yaml_path_var.set(file_path)
            # 自动加载YAML
            self.load_yaml_config(file_path)

    def load_yaml_config(self, yaml_path):
        """从YAML文件加载配置和类别"""
        if not self.yaml_available:
            return False

        try:
            import yaml
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 读取类别 - 支持多种格式
            new_classes = []
            if 'names' in config:
                names = config['names']
                if isinstance(names, dict):
                    # 格式: names: {0: 'person', 1: 'car', ...}
                    new_classes = [names[i] for i in sorted(names.keys())]
                elif isinstance(names, list):
                    # 格式: names: ['person', 'car', ...]
                    new_classes = names

            # 读取路径
            if 'path' in config:
                base_path = config['path']
                # 尝试自动设置图片和标签路径
                if 'train' in config:
                    train_path = config['train']
                    if train_path.endswith('/images'):
                        img_dir = os.path.join(base_path, train_path)
                        label_dir = os.path.join(base_path, train_path.replace('/images', '/labels'))
                        if os.path.exists(img_dir):
                            self.image_path_var.set(img_dir)
                        if os.path.exists(label_dir):
                            self.label_path_var.set(label_dir)
                    elif train_path.endswith('/labels'):
                        label_dir = os.path.join(base_path, train_path)
                        img_dir = os.path.join(base_path, train_path.replace('/labels', '/images'))
                        if os.path.exists(img_dir):
                            self.image_path_var.set(img_dir)
                        if os.path.exists(label_dir):
                            self.label_path_var.set(label_dir)
                    else:
                        # 尝试查找对应的images/labels目录
                        abs_train = os.path.join(base_path, train_path)
                        if os.path.exists(abs_train):
                            self.image_path_var.set(abs_train)
                            # 查找标签目录
                            possible_label_dir = abs_train.replace('/images', '/labels')
                            if os.path.exists(possible_label_dir):
                                self.label_path_var.set(possible_label_dir)

            # 更新类别
            if new_classes:
                self.class_names = new_classes
                # 重新生成颜色
                import random
                self.class_colors = {}
                for i, cls in enumerate(self.class_names):
                    # 使用预定义颜色或生成随机颜色
                    if cls not in self.class_colors:
                        self.class_colors[cls] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
                self.status_var.set(f"已从YAML加载 {len(new_classes)} 个类别")
                return True
            else:
                self.status_var.set("YAML文件中未找到类别信息")
                return False

        except Exception as e:
            messagebox.showerror("YAML加载错误", f"加载YAML文件失败: {str(e)}")
            return False

    def load_data(self):
        self.image_dir = self.image_path_var.get()
        self.label_dir = self.label_path_var.get()
        self.yaml_path = self.yaml_path_var.get()

        # 如果有YAML文件，尝试重新加载类别
        if self.yaml_path and os.path.exists(self.yaml_path):
            self.load_yaml_config(self.yaml_path)

        if not self.image_dir:
            messagebox.showwarning("警告", "请选择图片路径")
            return

        if not os.path.exists(self.image_dir):
            messagebox.showerror("错误", "图片路径不存在")
            return

        # 支持的图片格式
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']
        self.image_files = []
        self.label_files = {}

        # 加载图片文件
        for ext in img_extensions:
            pattern = os.path.join(self.image_dir, f"*{ext}")
            files = glob.glob(pattern, recursive=False)
            self.image_files.extend(files)

        for ext in img_extensions:
            pattern = os.path.join(self.image_dir, f"*{ext.upper()}")
            files = glob.glob(pattern, recursive=False)
            self.image_files.extend(files)

        self.image_files = sorted(set(self.image_files))

        if not self.image_files:
            messagebox.showwarning("警告", "图片文件夹中没有找到支持的图片格式")
            return

        # 加载标签文件
        if self.label_dir and os.path.exists(self.label_dir):
            for img_path in self.image_files:
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(self.label_dir, img_name + ".txt")
                if os.path.exists(label_path):
                    self.label_files[img_path] = label_path

        self.current_index = 0
        self.update_file_list()
        self.show_current_image()
        self.status_var.set(f"已加载 {len(self.image_files)} 张图片，{len(self.label_files)} 个标签文件")

    def update_file_list(self):
        self.file_listbox.delete(0, tk.END)
        for i, img_path in enumerate(self.image_files):
            name = os.path.basename(img_path)
            has_label = img_path in self.label_files
            prefix = "[✓] " if has_label else "[ ] "
            self.file_listbox.insert(tk.END, prefix + name)

        self.file_listbox.selection_clear(0, tk.END)
        if self.image_files:
            self.file_listbox.selection_set(self.current_index)
            self.file_listbox.see(self.current_index)

    def get_image_info(self, img_path):
        """读取YOLO标签文件"""
        if img_path not in self.label_files:
            return [], None, None

        label_path = self.label_files[img_path]
        boxes = []

        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        boxes.append({
                            'class_id': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height,
                            'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                        })
        except Exception as e:
            print(f"读取标签文件错误: {e}")

        # 获取图片尺寸
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except:
            img_width, img_height = 1, 1

        return boxes, img_width, img_height

    def show_image(self, img_path):
        """显示图片和标注框"""
        self.canvas.delete("all")

        try:
            # 加载图片
            img = Image.open(img_path)
            original_width, original_height = img.size

            # 计算缩放比例
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width < 50 or canvas_height < 50:
                canvas_width, canvas_height = 800, 600

            scale_x = canvas_width / original_width
            scale_y = canvas_height / original_height
            scale = min(scale_x, scale_y) * 0.95

            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.current_image_tk = ImageTk.PhotoImage(img_resized)

            # 居中显示
            x_offset = (canvas_width - new_width) // 2
            y_offset = (canvas_height - new_height) // 2

            self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.current_image_tk)

            # 读取并绘制标注框
            boxes, img_w, img_h = self.get_image_info(img_path)

            for box in boxes:
                # YOLO格式转像素坐标
                x_center = box['x_center'] * img_w
                y_center = box['y_center'] * img_h
                box_w = box['width'] * img_w
                box_h = box['height'] * img_h

                # 转换为canvas坐标
                x1 = (x_center - box_w / 2) * scale + x_offset
                y1 = (y_center - box_h / 2) * scale + y_offset
                x2 = (x_center + box_w / 2) * scale + x_offset
                y2 = (y_center + box_h / 2) * scale + y_offset

                # 获取颜色
                color = self.class_colors.get(box['class_name'], '#00FF00')

                # 绘制框
                self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)

                # 绘制标签
                label_text = f"{box['class_name']} ({box['class_id']})"
                self.canvas.create_text(x1, y1 - 5, text=label_text, fill=color, anchor=tk.SW, font=("Arial", 10, "bold"))

            # 显示图片信息
            info_text = f"图片: {os.path.basename(img_path)} | 尺寸: {original_width}x{original_height} | 标注数: {len(boxes)}"
            self.canvas.create_text(10, 10, text=info_text, fill="white", anchor=tk.NW, font=("Arial", 10))

            self.status_var.set(f"当前: {os.path.basename(img_path)} | {len(boxes)} 个标注框")

        except Exception as e:
            self.canvas.create_text(canvas_width//2, canvas_height//2, text=f"加载图片失败: {str(e)}", fill="red")

    def show_current_image(self):
        if not self.image_files:
            return
        img_path = self.image_files[self.current_index]
        self.show_image(img_path)
        self.page_info_var.set(f"{self.current_index + 1} / {len(self.image_files)}")
        self.file_listbox.selection_clear(0, tk.END)
        self.file_listbox.selection_set(self.current_index)

    def show_selected_file(self):
        selection = self.file_listbox.curselection()
        if selection:
            self.current_index = selection[0]
            self.show_current_image()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.show_current_image()

    def first_image(self):
        self.current_index = 0
        self.show_current_image()

    def last_image(self):
        self.current_index = len(self.image_files) - 1
        self.show_current_image()

    def refresh_current(self):
        self.show_current_image()

    def search_image(self):
        search_text = self.search_var.get().strip().lower()
        if not search_text or not self.image_files:
            return

        for i, img_path in enumerate(self.image_files):
            name = os.path.basename(img_path).lower()
            if search_text in name:
                self.current_index = i
                self.show_current_image()
                self.status_var.set(f"找到: {os.path.basename(img_path)}")
                return

        messagebox.showinfo("搜索结果", f"未找到包含 '{search_text}' 的图片")

    def clear_search(self):
        self.search_var.set("")

    def show_classes(self):
        """显示当前类别列表"""
        info_window = tk.Toplevel(self.root)
        info_window.title("当前类别列表")
        info_window.geometry("400x500")

        text_widget = tk.Text(info_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        content = "当前类别 (ID: 名称):\n\n"
        for i, cls in enumerate(self.class_names):
            color = self.class_colors.get(cls, '#00FF00')
            content += f"{i:2d}: {cls}\n"

        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)

        ttk.Button(info_window, text="关闭", command=info_window.destroy).pack(pady=10)

    def edit_classes(self):
        """编辑类别名称"""
        edit_window = tk.Toplevel(self.root)
        edit_window.title("编辑类别")
        edit_window.geometry("400x500")

        ttk.Label(edit_window, text="类别列表 (每行一个类别):").pack(anchor=tk.W, padx=5, pady=5)

        text_widget = tk.Text(edit_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        text_widget.insert(tk.END, "\n".join(self.class_names))

        def save_classes():
            content = text_widget.get("1.0", tk.END)
            new_classes = [line.strip() for line in content.split("\n") if line.strip()]
            if new_classes:
                self.class_names = new_classes
                # 重新生成颜色
                import random
                self.class_colors = {}
                for cls in self.class_names:
                    self.class_colors[cls] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
                self.refresh_current()
                edit_window.destroy()
                messagebox.showinfo("成功", f"已更新 {len(new_classes)} 个类别")

        button_frame = ttk.Frame(edit_window)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(button_frame, text="保存", command=save_classes).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="取消", command=edit_window.destroy).pack(side=tk.LEFT, padx=5)

    def toggle_fullscreen(self):
        if self.is_fullscreen:
            self.exit_fullscreen()
        else:
            self.root.attributes('-fullscreen', True)
            self.is_fullscreen = True

    def exit_fullscreen(self):
        self.root.attributes('-fullscreen', False)
        self.is_fullscreen = False

    def on_canvas_resize(self, event):
        if hasattr(self, 'image_files') and self.image_files:
            # 延迟重绘，避免频繁刷新
            self.root.after(100, self.show_current_image)


def main():
    """主函数，带错误处理"""
    try:
        root = tk.Tk()
        app = YoloViewer(root)
        root.mainloop()
    except Exception as e:
        # 捕获并显示错误
        import traceback
        error_msg = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        print(f"程序启动失败: {e}")
        print(error_msg)

        # 尝试显示错误对话框
        try:
            error_root = tk.Tk()
            error_root.withdraw()
            messagebox.showerror(
                "启动错误",
                f"程序启动失败！\n\n错误: {e}\n\n请检查是否已安装所有依赖:\npip install -r requirements.txt"
            )
            error_root.destroy()
        except:
            pass


if __name__ == "__main__":
    main()
