#!/usr/bin/env python3
"""
最佳MLP模型 - 学生学业表现预测
配置: (128,64,32), 权重: Dropout=1.0, Enrolled=1.5, Graduate=0.8
"""

import numpy as np
import pandas as pd
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(file_path='dataset/data.csv'):
    """加载数据集"""
    print("=" * 60)
    print("加载数据")
    print("=" * 60)

    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件不存在 - {file_path}")
        return None, None, None

    print(f"从 {file_path} 加载数据...")

    try:
        df = pd.read_csv(file_path, sep=';', encoding='utf-8')
    except:
        try:
            df = pd.read_csv(file_path, sep=';', encoding='latin1')
        except:
            df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8')

    # 清理列名
    df.columns = df.columns.str.replace('"', '').str.strip()
    df.columns = df.columns.str.replace('\t', '')

    target_col = 'Target'
    if target_col not in df.columns:
        possible_targets = ['Target', 'target', 'TARGET']
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break

    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    # 编码目标变量
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"类别映射: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # 处理数值列
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = X[col].astype(str).str.replace(',', '.').astype(float)
            except:
                pass

    # 处理分类特征
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols)

    # 处理缺失值
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())

    X = X.astype(float)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"数据加载完成: {X_scaled.shape[0]} 样本, {X_scaled.shape[1]} 特征")
    print(f"类别分布: Dropout={np.bincount(y)[0]}, Enrolled={np.bincount(y)[1]}, Graduate={np.bincount(y)[2]}")

    return X_scaled, y, le, scaler


def plot_class_distribution(y_train, y_test, le, save_dir='figures_best'):
    """绘制类别分布图"""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 训练集
    train_counts = np.bincount(y_train)
    bars1 = ax1.bar(le.classes_, train_counts, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax1.set_title('训练集类别分布', fontsize=12, fontweight='bold')
    ax1.set_ylabel('样本数量', fontsize=10)
    for bar, count in zip(bars1, train_counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 str(count), ha='center', va='bottom', fontsize=10)

    # 测试集
    test_counts = np.bincount(y_test)
    bars2 = ax2.bar(le.classes_, test_counts, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax2.set_title('测试集类别分布', fontsize=12, fontweight='bold')
    ax2.set_ylabel('样本数量', fontsize=10)
    for bar, count in zip(bars2, test_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(count), ha='center', va='bottom', fontsize=10)

    plt.suptitle('数据集类别分布', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 类别分布图已保存: {save_dir}/class_distribution.png")


def plot_training_curves(mlp, save_dir='figures_best'):
    """绘制训练曲线"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    if hasattr(mlp, 'loss_curve_') and mlp.loss_curve_ and len(mlp.loss_curve_) > 0:
        axes[0].plot(mlp.loss_curve_, 'b-', linewidth=2)
        axes[0].set_xlabel('迭代次数', fontsize=12)
        axes[0].set_ylabel('损失值', fontsize=12)
        axes[0].set_title('训练损失曲线', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        final_loss = mlp.loss_curve_[-1]
        axes[0].axhline(y=final_loss, color='red', linestyle='--', alpha=0.7)
        axes[0].text(len(mlp.loss_curve_) * 0.7, final_loss * 1.05,
                     f'最终损失: {final_loss:.4f}', fontsize=9, color='red')
    else:
        axes[0].text(0.5, 0.5, '无损失曲线数据', ha='center', va='center', fontsize=12)
        axes[0].set_title('训练损失曲线', fontsize=12, fontweight='bold')

    # 验证分数曲线
    if hasattr(mlp, 'validation_scores_') and mlp.validation_scores_ and len(mlp.validation_scores_) > 0:
        axes[1].plot(mlp.validation_scores_, 'g-', linewidth=2)
        axes[1].set_xlabel('迭代次数', fontsize=12)
        axes[1].set_ylabel('验证分数', fontsize=12)
        axes[1].set_title('验证分数曲线', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        best_score = max(mlp.validation_scores_)
        best_iter = np.argmax(mlp.validation_scores_)
        axes[1].plot(best_iter, best_score, 'ro', markersize=8)
        axes[1].annotate(f'最佳: {best_score:.4f}',
                         xy=(best_iter, best_score),
                         xytext=(best_iter + 5, best_score - 0.02),
                         fontsize=10,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    else:
        axes[1].text(0.5, 0.5, '无验证分数数据', ha='center', va='center', fontsize=12)
        axes[1].set_title('验证分数曲线', fontsize=12, fontweight='bold')

    plt.suptitle(f'训练过程 (迭代次数: {mlp.n_iter_})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 训练曲线已保存: {save_dir}/training_curves.png")


def plot_confusion_matrix_custom(y_test, y_pred, le, save_dir='figures_best'):
    """绘制混淆矩阵"""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, cmap='Blues', values_format='d')

    ax.set_title('最佳MLP模型 - 混淆矩阵', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存: {save_dir}/confusion_matrix.png")


def plot_performance_metrics(y_test, y_pred, le, save_dir='figures_best'):
    """绘制各类别性能指标"""
    os.makedirs(save_dir, exist_ok=True)

    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(le.classes_))
    width = 0.25

    bars1 = ax.bar(x - width, precision, width, label='精确率', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='召回率', color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1分数', color='#e74c3c')

    ax.set_xlabel('类别', fontsize=12)
    ax.set_ylabel('分数', fontsize=12)
    ax.set_title('最佳MLP模型 - 各类别性能指标', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(le.classes_)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 性能指标图已保存: {save_dir}/performance_metrics.png")


def plot_roc_curves(mlp, X_test, y_test, le, save_dir='figures_best'):
    """绘制ROC曲线（一对多）"""
    os.makedirs(save_dir, exist_ok=True)

    from sklearn.preprocessing import label_binarize

    # 二值化标签
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = 3

    # 获取预测概率
    y_score = mlp.predict_proba(X_test)

    # 计算每条ROC曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

    for i, color, class_name in zip(range(n_classes), colors, le.classes_):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc[i]:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='随机猜测')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假阳性率', fontsize=12)
    ax.set_ylabel('真阳性率', fontsize=12)
    ax.set_title('最佳MLP模型 - ROC曲线', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC曲线已保存: {save_dir}/roc_curves.png")


def plot_model_structure(save_dir='figures_best'):
    """绘制模型结构图"""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # 网络结构
    layers = ['输入层\n36个特征', '隐藏层1\n128个神经元', '隐藏层2\n64个神经元', '隐藏层3\n32个神经元',
              '输出层\n3个类别']

    x_positions = [1, 3.5, 5.5, 7.5, 9]
    y_position = 2.5

    for i, (layer, x) in enumerate(zip(layers, x_positions)):
        # 绘制节点圆
        circle = plt.Circle((x, y_position), 0.4, color='#3498db', ec='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, y_position - 0.6, layer, ha='center', va='top', fontsize=9, wrap=True)

        # 绘制连接线
        if i < len(x_positions) - 1:
            for j in range(3):  # 几条示意连线
                start_y = y_position - 0.2 + j * 0.2
                end_y = y_position - 0.2 + j * 0.2
                ax.plot([x + 0.4, x_positions[i + 1] - 0.4], [start_y, end_y],
                        'gray', linewidth=0.5, alpha=0.5)

    ax.text(5, 4.5, '最佳MLP网络结构', ha='center', fontsize=14, fontweight='bold')
    ax.text(5, 0.5, f'总参数量: 约 {(36 * 128 + 128) + (128 * 64 + 64) + (64 * 32 + 32) + (32 * 3 + 3):,}',
            ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_structure.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 模型结构图已保存: {save_dir}/model_structure.png")


def train_best_model(X, y, le, save_dir='figures_best'):
    """训练最佳MLP模型"""

    print("\n" + "=" * 60)
    print("最佳MLP模型训练")
    print("=" * 60)

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")

    # 绘制类别分布
    plot_class_distribution(y_train, y_test, le, save_dir)

    # 最佳配置
    best_config = {
        'hidden_layer_sizes': (128, 64, 32),
        'alpha': 0.0001,
        'learning_rate_init': 0.0005,
        'batch_size': 32,
        'activation': 'tanh',
        'solver': 'adam',
        'max_iter': 500,
        'early_stopping': True,
        'n_iter_no_change': 15,
        'validation_fraction': 0.1,
        'random_state': 42,
        'verbose': True
    }

    # 类别权重
    class_weights = {0: 1.0, 1: 1.5, 2: 0.8}
    sample_weight = np.array([class_weights[cls] for cls in y_train])

    print("\n模型配置:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print(f"  类别权重: {class_weights}")

    # 训练模型
    print("\n开始训练...")
    mlp = MLPClassifier(**best_config)
    mlp.fit(X_train, y_train, sample_weight=sample_weight)

    # 预测
    y_pred = mlp.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = mlp.score(X_test, y_test)

    print(f"\n训练结果:")
    print(f"  实际迭代次数: {mlp.n_iter_}")
    print(f"  最终损失: {mlp.loss_:.6f}")
    print(f"  测试集准确率: {accuracy:.4f}")
    print(f"  测试集加权F1: {f1:.4f}")

    # 绘制图表
    print("\n生成图表...")
    plot_training_curves(mlp, save_dir)
    plot_confusion_matrix_custom(y_test, y_pred, le, save_dir)
    plot_performance_metrics(y_test, y_pred, le, save_dir)
    plot_roc_curves(mlp, X_test, y_test, le, save_dir)
    plot_model_structure(save_dir)

    # 详细报告
    print("\n" + "=" * 60)
    print("详细分类报告")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=['Dropout', 'Enrolled', 'Graduate']))

    return mlp, f1, y_pred, y_test


def main():
    print("=" * 60)
    print("最佳MLP模型 - 学生学业表现预测")
    print("=" * 60)

    # 加载数据
    X, y, le, scaler = load_data('dataset/data.csv')

    if X is None:
        print("数据加载失败")
        return

    # 训练最佳模型
    save_dir = 'figures_best'
    mlp, f1, y_pred, y_test = train_best_model(X, y, le, save_dir)

    # 保存模型（可选）
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(mlp, 'models/best_mlp_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\n✅ 模型已保存到 models/best_mlp_model.pkl")

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"最佳配置: (128, 64, 32), alpha=0.0001, lr=0.0005")
    print(f"类别权重: Dropout=1.0, Enrolled=1.5, Graduate=0.8")
    print(f"最终F1分数: {f1:.4f}")

    print(f"\n生成的图表保存在: {save_dir}/")
    print("  - class_distribution.png (类别分布)")
    print("  - training_curves.png (训练曲线)")
    print("  - confusion_matrix.png (混淆矩阵)")
    print("  - performance_metrics.png (性能指标)")
    print("  - roc_curves.png (ROC曲线)")
    print("  - model_structure.png (模型结构)")


if __name__ == "__main__":
    main()