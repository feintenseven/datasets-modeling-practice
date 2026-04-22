#!/usr/bin/env python3
"""
SMOTE过采样 + MLP 完整流程（无Lasso）
直接解决类别不平衡问题
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
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
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
    print(f"原始类别分布: {np.bincount(y)}")

    return X_scaled, y, le


def apply_smote(X_train, y_train, sampling_strategy=None):
    """
    应用SMOTE过采样
    """
    print("\n" + "=" * 60)
    print("SMOTE过采样")
    print("=" * 60)

    # 显示原始类别分布
    unique, counts = np.unique(y_train, return_counts=True)
    print("原始训练集类别分布:")
    for cls, count in zip(unique, counts):
        cls_name = ['Dropout', 'Enrolled', 'Graduate'][cls]
        print(f"  {cls_name}: {count} 样本")

    # 默认采样策略：使所有类别样本数相同
    if sampling_strategy is None:
        sampling_strategy = 'auto'

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 显示过采样后的类别分布
    unique, counts = np.unique(y_train_resampled, return_counts=True)
    print("\nSMOTE后训练集类别分布:")
    for cls, count in zip(unique, counts):
        cls_name = ['Dropout', 'Enrolled', 'Graduate'][cls]
        print(f"  {cls_name}: {count} 样本")

    print(f"\n总样本数变化: {len(y_train)} -> {len(y_train_resampled)}")
    print(f"增加样本数: {len(y_train_resampled) - len(y_train)}")

    return X_train_resampled, y_train_resampled, smote


def plot_class_distribution_before_after(y_train, y_train_resampled, le, save_dir='figures_smote'):
    """绘制SMOTE前后的类别分布对比图"""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # SMOTE前
    before_counts = np.bincount(y_train)
    bars1 = ax1.bar(le.classes_, before_counts, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax1.set_title('SMOTE前', fontsize=12, fontweight='bold')
    ax1.set_ylabel('样本数量', fontsize=10)
    for bar, count in zip(bars1, before_counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 str(count), ha='center', va='bottom', fontsize=10)

    # SMOTE后
    after_counts = np.bincount(y_train_resampled)
    bars2 = ax2.bar(le.classes_, after_counts, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax2.set_title('SMOTE后', fontsize=12, fontweight='bold')
    ax2.set_ylabel('样本数量', fontsize=10)
    for bar, count in zip(bars2, after_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 str(count), ha='center', va='bottom', fontsize=10)

    plt.suptitle('SMOTE过采样前后类别分布对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'smote_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ SMOTE对比图已保存: {save_dir}/smote_comparison.png")


def train_mlp_with_smote(X_train, y_train, X_test, y_test, save_dir='figures_smote'):
    """训练使用SMOTE后的MLP模型"""
    print("\n" + "=" * 60)
    print("训练MLP模型（SMOTE过采样后）")
    print("=" * 60)

    # MLP配置
    mlp_config = {
        'hidden_layer_sizes': (64, 32, 16),
        'alpha': 0.0005,
        'learning_rate_init': 0.001,
        'batch_size': 32,
        'max_iter': 500,
        'early_stopping': True,
        'n_iter_no_change': 15,
        'validation_fraction': 0.1,
        'activation': 'relu',
        'solver': 'adam',
        'random_state': 42,
        'verbose': False
    }

    print("MLP配置:")
    for key, value in mlp_config.items():
        print(f"  {key}: {value}")

    mlp = MLPClassifier(**mlp_config)

    # SMOTE后数据已平衡，不需要sample_weight
    mlp.fit(X_train, y_train)

    # 预测和评估
    y_pred = mlp.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = mlp.score(X_test, y_test)

    print(f"\n训练结果:")
    print(f"  实际迭代次数: {mlp.n_iter_}")
    print(f"  最终损失: {mlp.loss_:.6f}")
    print(f"  测试集准确率: {accuracy:.4f}")
    print(f"  测试集加权F1: {f1:.4f}")

    return mlp, y_pred, f1


def plot_training_curves(mlp, save_dir='figures_smote'):
    """绘制训练曲线"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    if hasattr(mlp, 'loss_curve_') and mlp.loss_curve_ and len(mlp.loss_curve_) > 0:
        axes[0].plot(mlp.loss_curve_, 'b-', linewidth=2)
        axes[0].set_xlabel('迭代次数', fontsize=12)
        axes[0].set_ylabel('损失值', fontsize=12)
        axes[0].set_title('MLP训练损失曲线', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        final_loss = mlp.loss_curve_[-1]
        axes[0].axhline(y=final_loss, color='red', linestyle='--', alpha=0.7)
        axes[0].text(len(mlp.loss_curve_) * 0.7, final_loss * 1.05,
                     f'最终损失: {final_loss:.4f}', fontsize=9, color='red')

    # 验证分数曲线
    if hasattr(mlp, 'validation_scores_') and mlp.validation_scores_ and len(mlp.validation_scores_) > 0:
        axes[1].plot(mlp.validation_scores_, 'g-', linewidth=2)
        axes[1].set_xlabel('迭代次数', fontsize=12)
        axes[1].set_ylabel('验证分数', fontsize=12)
        axes[1].set_title('MLP验证分数曲线', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        best_score = max(mlp.validation_scores_)
        best_iter = np.argmax(mlp.validation_scores_)
        axes[1].plot(best_iter, best_score, 'ro', markersize=8)
        axes[1].annotate(f'最佳: {best_score:.4f}',
                         xy=(best_iter, best_score),
                         xytext=(best_iter + 3, best_score - 0.03),
                         fontsize=10)

    plt.suptitle(f'训练过程分析 (迭代次数: {mlp.n_iter_})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 训练曲线已保存: {save_dir}/training_curves.png")


def plot_confusion_matrix_custom(y_test, y_pred, le, save_dir='figures_smote'):
    """绘制混淆矩阵"""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, cmap='Blues', values_format='d')

    ax.set_title('MLP预测混淆矩阵 (SMOTE处理后)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存: {save_dir}/confusion_matrix.png")


def plot_performance_metrics(y_test, y_pred, le, save_dir='figures_smote'):
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
    ax.set_title('各类别性能指标 (SMOTE处理后)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(le.classes_)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 性能指标图已保存: {save_dir}/performance_metrics.png")


def print_detailed_report(y_test, y_pred, le):
    """打印详细的分类报告"""
    print("\n" + "=" * 60)
    print("详细分类报告 (SMOTE处理后)")
    print("=" * 60)

    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print(report)


def main():
    """主函数"""
    print("=" * 60)
    print("SMOTE过采样 + MLP 完整流程（无Lasso）")
    print("=" * 60)

    # 1. 加载数据
    X, y, le = load_data('dataset/data.csv')

    if X is None:
        print("数据加载失败")
        return

    # 2. 划分数据集
    print("\n" + "=" * 60)
    print("划分数据集")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")

    # 3. SMOTE过采样
    X_train_balanced, y_train_balanced, smote = apply_smote(
        X_train, y_train, sampling_strategy='auto'
    )

    # 4. 创建保存目录
    save_dir = 'figures_smote_only'
    os.makedirs(save_dir, exist_ok=True)

    # 5. 绘制SMOTE前后对比
    plot_class_distribution_before_after(y_train, y_train_balanced, le, save_dir)

    # 6. 训练MLP
    mlp, y_pred, f1 = train_mlp_with_smote(
        X_train_balanced, y_train_balanced, X_test, y_test, save_dir
    )

    # 7. 可视化
    plot_training_curves(mlp, save_dir)
    plot_confusion_matrix_custom(y_test, y_pred, le, save_dir)
    plot_performance_metrics(y_test, y_pred, le, save_dir)

    # 8. 打印详细报告
    print_detailed_report(y_test, y_pred, le)

    # 9. 总结
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)
    print(f"原始特征数: {X.shape[1]}")
    print(f"SMOTE后训练集大小: {len(y_train_balanced)}")
    print(f"MLP结构: {mlp.hidden_layer_sizes}")
    print(f"最终测试集F1分数: {f1:.4f}")

    print(f"\n生成的图表保存在: {save_dir}/")
    print("  - smote_comparison.png (SMOTE前后对比)")
    print("  - training_curves.png (训练曲线)")
    print("  - confusion_matrix.png (混淆矩阵)")
    print("  - performance_metrics.png (性能指标)")


if __name__ == "__main__":
    main()