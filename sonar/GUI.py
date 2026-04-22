#!/usr/bin/env python3
"""
声纳信号分类GUI - 美化版
使用训练好的88.10%准确率模型
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import joblib
import json
import os
import sys
from datetime import datetime
import random

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class SonarClassifierGUI:
    """声纳信号分类器GUI - 美化版"""

    def __init__(self, root):
        self.root = root
        self.root.title("⚡ 声纳信号分类器 - 岩石/矿石识别 ⚡")
        self.root.geometry("1100x800")
        self.root.configure(bg='#1a1a2e')

        # 先初始化状态变量
        self.status_var = tk.StringVar(value="🚀 初始化中...")

        # 初始化其他变量
        self.model = None
        self.scaler = None
        self.model_info = None
        self.feature_entries = []
        self.animation_id = None

        # 创建UI（先创建UI组件）
        self.create_widgets()

        # 然后加载模型
        self.load_model()

    def setup_styles(self):
        """设置样式"""
        style = ttk.Style()
        style.theme_use('clam')

        # 自定义进度条样式
        style.configure('danger.Horizontal.TProgressbar',
                        background='#ff4444', troughcolor='#330000')
        style.configure('warning.Horizontal.TProgressbar',
                        background='#ffaa44', troughcolor='#332200')
        style.configure('info.Horizontal.TProgressbar',
                        background='#44ff44', troughcolor='#003300')

        # 自定义按钮样式
        style.configure('Danger.TButton',
                        background='#ff4444', foreground='white',
                        font=('Arial', 12, 'bold'))
        style.configure('Success.TButton',
                        background='#44ff44', foreground='black',
                        font=('Arial', 12, 'bold'))

    def load_model(self):
        """加载保存的模型"""
        try:
            # 加载模型
            model_path = 'models/best_mlp_model.pkl'
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print("✓ 模型加载成功")
                self.status_var.set("✓ 模型加载成功 - 就绪")
            else:
                self.status_var.set("❌ 模型文件不存在")
                messagebox.showerror("错误", f"模型文件不存在: {model_path}\n请先运行 save_complete_model.py")
                return

            # 加载标准化器
            scaler_path = 'models/scaler.pkl'
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("✓ 标准化器加载成功")

            # 加载模型信息
            info_path = 'models/model_info.json'
            if os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    self.model_info = json.load(f)
                acc = self.model_info['performance']['test_accuracy']
                print(f"✓ 模型信息加载成功 (准确率: {acc:.2%})")
                self.status_var.set(f"✓ 模型加载成功 - 准确率: {acc:.1%}")

            # 更新准确率显示
            self.update_accuracy_display()

            # 更新模型信息选项卡
            self.update_info_tab()

        except Exception as e:
            print(f"加载模型失败: {e}")
            self.status_var.set("❌ 模型加载失败")
            messagebox.showerror("错误", f"加载模型失败: {e}")

    def update_accuracy_display(self):
        """更新准确率显示"""
        if hasattr(self, 'acc_label') and self.model_info:
            acc = self.model_info['performance']['test_accuracy']
            self.acc_label.config(text=f"🎯 模型准确率: {acc:.1%}")

    def update_info_tab(self):
        """更新模型信息选项卡内容"""
        if hasattr(self, 'info_text') and self.model_info:
            info = self.get_model_info_text()
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete('1.0', tk.END)
            self.info_text.insert('1.0', info)
            self.info_text.config(state=tk.DISABLED)

    def get_model_info_text(self):
        """获取模型信息文本"""
        if not self.model_info:
            return """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                              模型信息未加载                                   ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    请先运行 save_complete_model.py 训练并保存模型。

    运行命令：
        python save_complete_model.py

    然后重新启动此GUI应用程序。
    """

        info = f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                            模型详细信息                                      ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    【模型配置】
      网络结构: {self.model_info['config']['hidden_layer_sizes']}
      正则化参数 (alpha): {self.model_info['config']['alpha']}
      批次大小 (batch_size): {self.model_info['config']['batch_size']}
      初始学习率: {self.model_info['config']['learning_rate_init']}
      激活函数: {self.model_info['config']['activation']}
      优化器: {self.model_info['config']['solver']}
      早停: {self.model_info['config']['early_stopping']}
      最大迭代次数: {self.model_info['config']['max_iter']}
      早停耐心: {self.model_info['config']['n_iter_no_change']}
      随机种子: {self.model_info['config']['random_state']}

    【模型性能】
      训练准确率: {self.model_info['performance']['train_accuracy']:.2%}
      测试准确率: {self.model_info['performance']['test_accuracy']:.2%}
      实际迭代次数: {self.model_info['performance']['n_iter']}
      最终损失: {self.model_info['performance']['final_loss']:.6f}

    【模型结构】
      输入特征数: {self.model_info['model_parameters']['n_features_in']}
      输出类别: {self.model_info['classes']}
      总层数: {self.model_info['model_parameters']['n_layers']}

    【权重矩阵形状】
    """
        for i, coef in enumerate(self.model_info['model_parameters']['coefs']):
            shape = np.array(coef).shape
            info += f"  第{i}层: {shape}\n"

        info += f"""
    【训练数据】
      数据集: Sonar 数据集
      样本数量: 208
      特征数量: 60
      类别: 0=岩石(Rock) 🪨, 1=矿石(Mine) 💣
      数据分布: Rock=97, Mine=111

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                              使用说明                                        ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    1. 🔍 单个预测：
       - 在"单个预测"选项卡中输入60个特征值
       - 点击"🔮 开始预测"按钮获取结果
       - 可以点击"📊 加载示例数据"加载真实样本测试

    2. 📁 批量预测：
       - 准备CSV文件（60个特征列，无表头）
       - 点击"浏览"选择文件
       - 点击"开始预测"查看批量结果

    3. 💡 特征值说明：
       - 特征值范围通常在0-1之间（原始数据）
       - 60个特征对应声纳信号的60个频段能量值
       - 点击加载示例数据会自动填充正确的原始数据

    4. 🎨 视觉效果：
       - 检测到矿石 💣：红色闪烁、窗口震动、惊恐表情
       - 检测到岩石 🪨：绿色显示、安全表情
    """
        return info

    def create_widgets(self):
        """创建UI组件"""

        # 标题栏
        title_frame = tk.Frame(self.root, bg='#16213e', height=90)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)

        title_label = tk.Label(title_frame, text="⚡ 声纳信号分类器 ⚡",
                               font=('Arial', 26, 'bold'),
                               fg='#ffd700', bg='#16213e')
        title_label.pack(pady=15)

        subtitle_label = tk.Label(title_frame, text="岩石 🪨 vs 矿石 💣 | 准确率 88.1%",
                                  font=('Arial', 11),
                                  fg='#88aaff', bg='#16213e')
        subtitle_label.pack()

        # 准确率标签
        self.acc_label = tk.Label(title_frame, text="🎯 模型准确率: 加载中...",
                                  font=('Arial', 11, 'bold'),
                                  fg='#ffd700', bg='#16213e')
        self.acc_label.place(x=850, y=55)

        # 主框架
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # 创建Notebook（选项卡）
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True)

        # 选项卡1：单个预测
        self.create_prediction_tab(notebook)

        # 选项卡2：批量预测
        self.create_batch_tab(notebook)

        # 选项卡3：模型信息
        self.create_info_tab(notebook)

        # 底部状态栏
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                              bd=1, relief=tk.SUNKEN, anchor=tk.W,
                              font=('Arial', 10), bg='#2d2d44', fg='#ffffff')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_prediction_tab(self, notebook):
        """创建单个预测选项卡"""
        tab = tk.Frame(notebook, bg='#1a1a2e')
        notebook.add(tab, text="🔍 单个预测")

        # 左右分栏
        left_frame = tk.Frame(tab, bg='#1a1a2e')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        right_frame = tk.Frame(tab, bg='#1a1a2e', width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)
        right_frame.pack_propagate(False)

        # 左侧：特征输入区域
        input_frame = tk.LabelFrame(left_frame, text="📊 输入60个特征值",
                                    font=('Arial', 12, 'bold'),
                                    bg='#1a1a2e', fg='#ffd700')
        input_frame.pack(fill=tk.BOTH, expand=True)

        # 创建滚动框架
        canvas = tk.Canvas(input_frame, bg='#1a1a2e', highlightthickness=0)
        scrollbar = tk.Scrollbar(input_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#1a1a2e')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 创建60个输入框（每行10个）
        for i in range(60):
            row = i // 10
            col = i % 10

            label = tk.Label(scrollable_frame, text=f"F{i + 1}:",
                             font=('Arial', 9), bg='#1a1a2e', fg='#88aaff')
            label.grid(row=row, column=col * 2, padx=2, pady=2, sticky='e')

            entry = tk.Entry(scrollable_frame, width=8, font=('Arial', 9),
                             bg='#2d2d44', fg='#ffffff', insertbackground='#ffffff')
            entry.grid(row=row, column=col * 2 + 1, padx=2, pady=2)
            entry.insert(0, "0.0")
            self.feature_entries.append(entry)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 按钮框架
        button_frame = tk.Frame(left_frame, bg='#1a1a2e')
        button_frame.pack(fill=tk.X, pady=10)

        # 预设值按钮
        tk.Button(button_frame, text="📊 加载示例数据", command=self.load_sample_data,
                  font=('Arial', 10, 'bold'), bg='#3498db', fg='white',
                  activebackground='#2980b9', padx=15, pady=5).pack(side=tk.LEFT, padx=5)

        tk.Button(button_frame, text="🧹 清空所有", command=self.clear_all_entries,
                  font=('Arial', 10, 'bold'), bg='#95a5a6', fg='white',
                  activebackground='#7f8c8d', padx=15, pady=5).pack(side=tk.LEFT, padx=5)

        # 右侧：结果显示区域
        result_frame = tk.LabelFrame(right_frame, text="🎯 预测结果",
                                     font=('Arial', 14, 'bold'),
                                     bg='#1a1a2e', fg='#ffd700')
        result_frame.pack(fill=tk.BOTH, expand=True)

        # 预测按钮
        predict_btn = tk.Button(result_frame, text="🔮 开始预测 🔮",
                                command=self.predict_single,
                                font=('Arial', 16, 'bold'),
                                bg='#27ae60', fg='white',
                                activebackground='#229954',
                                height=2, cursor='hand2')
        predict_btn.pack(fill=tk.X, padx=20, pady=20)

        # 结果显示
        self.result_label = tk.Label(result_frame, text="✨ 等待输入 ✨",
                                     font=('Arial', 18, 'bold'),
                                     bg='#1a1a2e', fg='#88aaff')
        self.result_label.pack(pady=20)

        self.prob_label = tk.Label(result_frame, text="",
                                   font=('Arial', 12),
                                   bg='#1a1a2e', fg='#88aaff')
        self.prob_label.pack(pady=10)

        # 置信度条
        self.confidence_bar = ttk.Progressbar(result_frame, length=280, mode='determinate',
                                              style='info.Horizontal.TProgressbar')
        self.confidence_bar.pack(pady=10)

        # 说明文字
        info_text = tk.Label(result_frame,
                             text="💡 使用说明：\n\n"
                                  "• 特征值范围通常为0-1之间\n"
                                  "• 输入60个特征值（每行10个）\n"
                                  "• 点击「加载示例数据」快速测试\n"
                                  "• 检测到矿石时会有震动和闪烁警告",
                             font=('Arial', 9), bg='#1a1a2e', fg='#88aaff',
                             justify=tk.LEFT)
        info_text.pack(pady=20)

    def create_batch_tab(self, notebook):
        """创建批量预测选项卡"""
        tab = tk.Frame(notebook, bg='#1a1a2e')
        notebook.add(tab, text="📁 批量预测")

        # 文件选择区域
        file_frame = tk.LabelFrame(tab, text="📂 选择数据文件",
                                   font=('Arial', 12, 'bold'),
                                   bg='#1a1a2e', fg='#ffd700')
        file_frame.pack(fill=tk.X, padx=20, pady=10)

        self.file_path_var = tk.StringVar()
        tk.Entry(file_frame, textvariable=self.file_path_var, width=60,
                 font=('Arial', 10), bg='#2d2d44', fg='#ffffff').pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(file_frame, text="📁 浏览", command=self.select_file,
                  font=('Arial', 10, 'bold'), bg='#3498db', fg='white',
                  activebackground='#2980b9', padx=20).pack(side=tk.LEFT, padx=5)
        tk.Button(file_frame, text="🚀 开始预测", command=self.predict_batch,
                  font=('Arial', 10, 'bold'), bg='#27ae60', fg='white',
                  activebackground='#229954', padx=20).pack(side=tk.LEFT, padx=5)

        # 结果显示区域
        result_frame = tk.LabelFrame(tab, text="📊 预测结果",
                                     font=('Arial', 12, 'bold'),
                                     bg='#1a1a2e', fg='#ffd700')
        result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 创建表格
        columns = ('序号', '岩石概率', '矿石概率', '预测结果', '置信度')
        self.tree = ttk.Treeview(result_frame, columns=columns, show='headings', height=15)

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)

        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 统计信息
        self.batch_stats_label = tk.Label(tab, text="", font=('Arial', 11),
                                          bg='#1a1a2e', fg='#ffd700')
        self.batch_stats_label.pack(pady=10)

    def create_info_tab(self, notebook):
        """创建模型信息选项卡"""
        tab = tk.Frame(notebook, bg='#1a1a2e')
        notebook.add(tab, text="ℹ️ 模型信息")

        self.info_text = tk.Text(tab, wrap=tk.WORD, font=('Consolas', 10),
                                 bg='#1a1a2e', fg='#88aaff',
                                 insertbackground='#ffffff')
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 先显示加载中的信息
        info = "📡 加载模型信息中..."
        self.info_text.insert('1.0', info)
        self.info_text.config(state=tk.DISABLED)

    def load_sample_data(self):
        """加载示例数据（矿石样本，原始数据）"""
        try:
            from preprocessing.scaler import DataPreprocessor
            preprocessor = DataPreprocessor()
            X, y, _ = preprocessor.load_data()

            # 找到第一个矿石样本（y==1）
            mine_indices = [i for i, label in enumerate(y) if label == 1]
            if mine_indices:
                sample = X[mine_indices[0]]  # 取第一个矿石样本
            else:
                sample = X[0]  # 如果没有矿石样本，取第一个

            for i, entry in enumerate(self.feature_entries):
                entry.delete(0, tk.END)
                entry.insert(0, f"{sample[i]:.6f}")

            self.status_var.set("💣 已加载矿石样本（危险！）")
        except Exception as e:
            self.status_var.set(f"❌ 加载示例数据失败: {e}")

    def clear_all_entries(self):
        """清空所有输入框"""
        for entry in self.feature_entries:
            entry.delete(0, tk.END)
            entry.insert(0, "0.0")
        self.result_label.config(text="✨ 等待输入 ✨", fg='#88aaff')
        self.prob_label.config(text="")
        self.confidence_bar['value'] = 0
        self.status_var.set("🧹 已清空所有输入")

    def predict_single(self):
        """单个预测 - 美化版"""
        if self.model is None:
            messagebox.showerror("错误", "模型未加载")
            return

        try:
            # 获取特征值
            features = []
            for entry in self.feature_entries:
                value = float(entry.get())
                features.append(value)

            features = np.array(features).reshape(1, -1)

            # 标准化
            if self.scaler:
                features = self.scaler.transform(features)

            # 预测
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]

            confidence = probabilities[prediction] * 100

            # ========== 美化部分 ==========
            if prediction == 1:  # 矿石（危险）
                # 危险表情随机组合
                danger_icons = [
                    "💣💥💀", "🤯⚠️🔥", "💢😱💣", "🎯💢⚠️", "🔥💣🤯",
                    "💀⚠️💣", "😱🔥💢", "💥🤯⚠️", "💣💀😱", "⚠️💢🔥"
                ]
                icon = random.choice(danger_icons)
                result_text = f"{icon} 危险！探测到矿石 (Mine)！{icon}"
                result_color = '#ff4444'
                # 添加闪烁效果
                self.flash_danger()
                # 添加震动效果
                self.shake_window()
                # 播放警告音（Windows）
                self.play_alert()
            else:  # 岩石（安全）
                # 安全表情随机组合
                safe_icons = [
                    "🪨✅😊", "👍🌊⛰️", "🪨✨✓", "🔵🪨✅", "😊⛰️👍",
                    "✅🪨🌊", "✨🔵🪨", "😊👍✅", "🪨💧✓", "🌊⛰️😊"
                ]
                icon = random.choice(safe_icons)
                result_text = f"{icon} 安全！探测到岩石 (Rock) {icon}"
                result_color = '#44ff44'

            # 更新结果标签
            self.result_label.config(text=result_text, fg=result_color,
                                     font=('Arial', 18, 'bold'))

            # 更新概率标签
            confidence_char = "💪" if confidence > 80 else "👍" if confidence > 60 else "🤔"
            self.prob_label.config(
                text=f"{confidence_char} 置信度: {confidence:.1f}%\n\n"
                     f"🪨 岩石概率: {probabilities[0]:.2%}\n"
                     f"💣 矿石概率: {probabilities[1]:.2%}",
                font=('Arial', 12)
            )

            # 更新置信度条颜色
            if confidence > 80:
                bar_style = 'danger.Horizontal.TProgressbar'
            elif confidence > 60:
                bar_style = 'warning.Horizontal.TProgressbar'
            else:
                bar_style = 'info.Horizontal.TProgressbar'

            self.confidence_bar['style'] = bar_style
            self.confidence_bar['value'] = confidence

            # 更新状态栏
            if prediction == 1:
                self.status_var.set(f"💢 危险警告！检测到矿石！置信度: {confidence:.1f}%")
            else:
                self.status_var.set(f"✅ 安全确认！检测到岩石！置信度: {confidence:.1f}%")

        except ValueError as e:
            messagebox.showerror("错误", f"输入值无效: {e}")
        except Exception as e:
            messagebox.showerror("错误", f"预测失败: {e}")

    def select_file(self):
        """选择文件"""
        file_path = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("CSV文件", "*.csv"), ("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)

    def predict_batch(self):
        """批量预测"""
        if self.model is None:
            messagebox.showerror("错误", "模型未加载")
            return

        file_path = self.file_path_var.get()
        if not file_path:
            messagebox.showwarning("警告", "请先选择文件")
            return

        try:
            # 读取文件
            data = np.loadtxt(file_path, delimiter=',')

            if data.ndim == 1:
                data = data.reshape(1, -1)

            if data.shape[1] != 60:
                messagebox.showerror("错误", f"特征数量错误: 需要60个特征，实际{data.shape[1]}个")
                return

            # 标准化
            if self.scaler:
                data = self.scaler.transform(data)

            # 预测
            predictions = self.model.predict(data)
            probabilities = self.model.predict_proba(data)

            # 清空之前的显示
            for item in self.tree.get_children():
                self.tree.delete(item)

            # 显示结果
            rock_count = 0
            mine_count = 0

            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                if pred == 1:
                    pred_label = "💣 矿石(Mine) 💣"
                    mine_count += 1
                else:
                    pred_label = "🪨 岩石(Rock) 🪨"
                    rock_count += 1

                confidence = prob[pred] * 100

                self.tree.insert('', 'end', values=(
                    i + 1,
                    f"{prob[0]:.2%}",
                    f"{prob[1]:.2%}",
                    pred_label,
                    f"{confidence:.1f}%"
                ))

            # 更新统计信息
            total = len(predictions)
            danger_icon = "⚠️" if mine_count > 0 else "✅"
            stats = f"{danger_icon} 统计: 总计 {total} 个样本 | 🪨 岩石: {rock_count} 个 | 💣 矿石: {mine_count} 个"
            self.batch_stats_label.config(text=stats)
            self.status_var.set(f"✅ 批量预测完成 - 处理了{total}个样本")

            # 如果检测到矿石，发出警告
            if mine_count > 0:
                self.flash_danger()
                self.play_alert()

        except Exception as e:
            messagebox.showerror("错误", f"批量预测失败: {e}")
            self.status_var.set(f"❌ 批量预测失败: {e}")

    # ========== 辅助美化方法 ==========

    def flash_danger(self):
        """危险闪烁效果"""

        def flash(count=0):
            if count >= 6:
                self.root.configure(bg='#1a1a2e')
                return
            color = '#8b0000' if count % 2 == 0 else '#1a1a2e'
            self.root.configure(bg=color)
            self.root.after(200, lambda: flash(count + 1))

        flash()

    def shake_window(self):
        """窗口震动效果"""
        x, y = self.root.winfo_x(), self.root.winfo_y()
        for i in range(5):
            offset = 8 if i % 2 == 0 else -8
            self.root.geometry(f"+{x + offset}+{y + offset}")
            self.root.update()
            self.root.after(30)
        self.root.geometry(f"+{x}+{y}")

    def play_alert(self):
        """播放警告音（Windows）"""
        try:
            import winsound
            import threading
            def beep():
                for _ in range(2):
                    winsound.Beep(1000, 200)
                    winsound.Beep(800, 200)

            threading.Thread(target=beep, daemon=True).start()
        except:
            pass  # 非Windows系统或没有winsound就跳过


def main():
    """主函数"""
    root = tk.Tk()
    app = SonarClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()