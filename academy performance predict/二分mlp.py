#!/usr/bin/env python3
"""
基础MLP - 二分类（毕业 vs 不毕业）
最纯粹的MLP，无花哨调参
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
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(file_path='dataset/data.csv'):
    """加载数据并转换为二分类"""
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
    y_original = df[target_col].copy()

    # 转换为二分类：毕业(Graduate)=1，其他(Dropout+Enrolled)=0
    y_binary = (y_original == 'Graduate').astype(int)

    print(f"原始三分类分布: {y_original.value_counts().to_dict()}")
    print(
        f"二分类分布: 不毕业={np.sum(y_binary == 0)} ({(1 - np.mean(y_binary)) * 100:.1f}%), 毕业={np.sum(y_binary == 1)} ({np.mean(y_binary) * 100:.1f}%)")

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

    return X_scaled, y_binary, scaler


def plot_binary_distribution(y_train, y_test, save_dir='figures_binary'):
    """绘制二分类分布图"""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 训练集
    train_counts = [np.sum(y_train == 0), np.sum(y_train == 1)]
    bars1 = ax1.bar(['不毕业', '毕业'], train_counts, color=['#ff6b6b', '#45b7d1'])
    ax1.set_title('训练集', fontsize=12, fontweight='bold')
    ax1.set_ylabel('样本数量', fontsize=10)
    for bar, count in zip(bars1, train_counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 str(count), ha='center', va='bottom', fontsize=10)

    # 测试集
    test_counts = [np.sum(y_test == 0), np.sum(y_test == 1)]
    bars2 = ax2.bar(['不毕业', '毕业'], test_counts, color=['#ff6b6b', '#45b7d1'])
    ax2.set_title('测试集', fontsize=12, fontweight='bold')
    ax2.set_ylabel('样本数量', fontsize=10)
    for bar, count in zip(bars2, test_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(count), ha='center', va='bottom', fontsize=10)

    plt.suptitle('二分类数据集分布（毕业 vs 不毕业）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'binary_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 分布图已保存: {save_dir}/binary_distribution.png")


def plot_training_curves(mlp, save_dir='figures_binary'):
    """绘制训练曲线"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    if hasattr(mlp, 'loss_curve_') and mlp.loss_curve_:
        axes[0].plot(mlp.loss_curve_, 'b-', linewidth=2)
        axes[0].set_xlabel('迭代次数', fontsize=12)
        axes[0].set_ylabel('损失值', fontsize=12)
        axes[0].set_title('训练损失曲线', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        final_loss = mlp.loss_curve_[-1]
        axes[0].axhline(y=final_loss, color='red', linestyle='--', alpha=0.7)
        axes[0].text(len(mlp.loss_curve_) * 0.7, final_loss * 1.05,
                     f'最终损失: {final_loss:.4f}', fontsize=9, color='red')

    # 验证分数曲线
    if hasattr(mlp, 'validation_scores_') and mlp.validation_scores_:
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
                         fontsize=10)

    plt.suptitle(f'训练过程 (迭代次数: {mlp.n_iter_})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 训练曲线已保存: {save_dir}/training_curves.png")


def plot_confusion_matrix_custom(y_test, y_pred, save_dir='figures_binary'):
    """绘制混淆矩阵"""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['不毕业', '毕业'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')

    ax.set_title('基础MLP - 混淆矩阵', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存: {save_dir}/confusion_matrix.png")


def plot_roc_curve(mlp, X_test, y_test, save_dir='figures_binary'):
    """绘制ROC曲线"""
    os.makedirs(save_dir, exist_ok=True)

    y_score = mlp.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#45b7d1', lw=2, label=f'MLP (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='随机猜测')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假阳性率', fontsize=12)
    ax.set_ylabel('真阳性率', fontsize=12)
    ax.set_title('基础MLP - ROC曲线', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC曲线已保存: {save_dir}/roc_curve.png")


def train_basic_mlp(X, y, save_dir='figures_binary'):
    """训练基础MLP（最简单配置）"""

    print("\n" + "=" * 60)
    print("基础MLP训练（二分类）")
    print("=" * 60)

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")

    # 绘制分布图
    plot_binary_distribution(y_train, y_test, save_dir)

    # 最简单的MLP配置
    print("\n模型配置（最简单）:")
    print("  hidden_layer_sizes: (100,)")
    print("  alpha: 0.0001")
    print("  random_state: 42")
    print("  max_iter: 200")

    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        alpha=0.0001,
        random_state=42,
        max_iter=200,
        verbose=True
    )

    print("\n开始训练...")
    mlp.fit(X_train, y_train)

    # 预测
    y_pred = mlp.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    accuracy = mlp.score(X_test, y_test)

    print(f"\n训练结果:")
    print(f"  实际迭代次数: {mlp.n_iter_}")
    print(f"  最终损失: {mlp.loss_:.6f}")
    print(f"  测试集准确率: {accuracy:.4f}")
    print(f"  测试集F1: {f1:.4f}")

    # 绘制图表
    print("\n生成图表...")
    plot_training_curves(mlp, save_dir)
    plot_confusion_matrix_custom(y_test, y_pred, save_dir)
    plot_roc_curve(mlp, X_test, y_test, save_dir)

    # 详细报告
    print("\n" + "=" * 60)
    print("详细分类报告")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=['不毕业', '毕业']))

    # 混淆矩阵数值
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n混淆矩阵:")
    print(f"           预测不毕业  预测毕业")
    print(f"实际不毕业:     {cm[0, 0]:4d}       {cm[0, 1]:4d}")
    print(f"实际毕业:       {cm[1, 0]:4d}       {cm[1, 1]:4d}")

    return mlp, f1, y_pred, y_test


def main():
    print("=" * 60)
    print("基础MLP - 二分类（毕业预测）")
    print("=" * 60)

    # 加载数据
    X, y, scaler = load_data('dataset/data.csv')

    if X is None:
        print("数据加载失败")
        return

    # 训练基础模型
    save_dir = 'figures_binary'
    mlp, f1, y_pred, y_test = train_basic_mlp(X, y, save_dir)

    # 保存模型
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(mlp, 'models/basic_mlp_binary.pkl')
    joblib.dump(scaler, 'models/scaler_binary.pkl')
    print("\n✅ 模型已保存到 models/basic_mlp_binary.pkl")

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"任务: 预测学生是否能毕业")
    print(f"模型: 基础MLP - 单隐藏层100个神经元")
    print(f"测试集F1分数: {f1:.4f}")
    print(f"测试集准确率: {np.mean(y_pred == y_test):.4f}")

    print(f"\n生成的图表保存在: {save_dir}/")
    print("  - binary_distribution.png (类别分布)")
    print("  - training_curves.png (训练曲线)")
    print("  - confusion_matrix.png (混淆矩阵)")
    print("  - roc_curve.png (ROC曲线)")


if __name__ == "__main__":
    main()