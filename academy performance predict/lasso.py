#!/usr/bin/env python3
"""
使用Lasso特征选择的MLP模型 - 完整实现
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
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
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
    print(f"类别分布: {np.bincount(y)}")

    return X_scaled, y, le


def get_sample_weights(y):
    """计算样本权重（处理类别不平衡）"""
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    sample_weight = np.zeros_like(y, dtype=float)
    for cls, w in zip(classes, weights):
        sample_weight[y == cls] = w
    return sample_weight


def lasso_feature_selection(X_train, y_train, X_test):
    """
    使用Lasso进行特征选择
    返回：选择后的训练集、测试集、以及选择器
    """
    print("\n" + "=" * 60)
    print("Lasso特征选择")
    print("=" * 60)

    # 使用LassoCV自动选择最佳正则化参数
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-4, 1, 50))
    lasso.fit(X_train, y_train)

    # 打印Lasso信息
    print(f"最佳alpha值: {lasso.alpha_:.6f}")
    print(f"非零系数数量: {np.sum(lasso.coef_ != 0)}")
    print(f"系数绝对值范围: [{np.abs(lasso.coef_).min():.6f}, {np.abs(lasso.coef_).max():.6f}]")

    # 特征选择（保留系数非零的特征）
    selector = SelectFromModel(lasso, threshold=1e-5, prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    # 获取被选中的特征索引
    selected_mask = selector.get_support()

    print(f"\n特征选择结果:")
    print(f"  原始特征数: {X_train.shape[1]}")
    print(f"  选中特征数: {X_train_selected.shape[1]}")
    print(f"  压缩率: {(1 - X_train_selected.shape[1] / X_train.shape[1]) * 100:.1f}%")

    return X_train_selected, X_test_selected, selector, lasso


def plot_lasso_coefficients(lasso, feature_names, save_dir='figures'):
    """绘制Lasso系数图"""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # 获取非零系数
    coef = lasso.coef_
    non_zero_idx = np.where(coef != 0)[0]
    non_zero_coef = coef[non_zero_idx]

    # 只显示前30个最重要的特征
    top_n = min(30, len(non_zero_idx))
    top_idx = np.argsort(np.abs(non_zero_coef))[-top_n:]

    colors = ['red' if c < 0 else 'green' for c in non_zero_coef[top_idx]]

    ax.barh(range(top_n), non_zero_coef[top_idx], color=colors)
    ax.set_xlabel('系数值', fontsize=12)
    ax.set_ylabel('特征索引', fontsize=12)
    ax.set_title(f'Lasso特征系数 (选中{len(non_zero_idx)}个特征)', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='正相关'),
                       Patch(facecolor='red', label='负相关')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lasso_coefficients.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Lasso系数图已保存: {save_dir}/lasso_coefficients.png")


def plot_feature_selection_comparison(original_dim, selected_dim, save_dir='figures'):
    """绘制特征选择前后对比图"""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 特征数量对比
    categories = ['原始特征', 'Lasso选择后']
    counts = [original_dim, selected_dim]
    colors = ['#3498db', '#2ecc71']

    bars = ax1.bar(categories, counts, color=colors, edgecolor='black')
    ax1.set_ylabel('特征数量', fontsize=12)
    ax1.set_title('特征数量对比', fontsize=12, fontweight='bold')

    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(count), ha='center', va='bottom', fontsize=11)

    # 压缩率饼图
    compression_rate = (original_dim - selected_dim) / original_dim * 100
    sizes = [selected_dim, original_dim - selected_dim]
    labels = [f'保留特征\n{selected_dim}个', f'去除特征\n{original_dim - selected_dim}个']
    colors_pie = ['#2ecc71', '#e74c3c']

    ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'特征压缩率: {compression_rate:.1f}%', fontsize=12, fontweight='bold')

    plt.suptitle('Lasso特征选择效果', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_selection_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 特征选择对比图已保存: {save_dir}/feature_selection_comparison.png")


def train_mlp_with_lasso(X_train, y_train, X_test, y_test, save_dir='figures'):
    """
    训练使用Lasso特征选择后的MLP模型
    """
    print("\n" + "=" * 60)
    print("训练MLP模型（Lasso特征选择后）")
    print("=" * 60)

    # 计算样本权重
    sample_weight = get_sample_weights(y_train)

    # MLP配置（针对Lasso后的特征优化）
    mlp_config = {
        'hidden_layer_sizes': (64, 32, 16),  # 三层网络
        'alpha': 0.0005,  # 正则化系数
        'learning_rate_init': 0.001,  # 学习率
        'batch_size': 32,  # 批量大小
        'max_iter': 500,  # 最大迭代次数
        'early_stopping': True,  # 早停
        'n_iter_no_change': 15,  # 早停耐心值
        'validation_fraction': 0.1,  # 验证集比例
        'activation': 'relu',  # 激活函数
        'solver': 'adam',  # 优化器
        'random_state': 42,  # 随机种子
        'verbose': False
    }

    print("MLP配置:")
    for key, value in mlp_config.items():
        print(f"  {key}: {value}")

    # 创建并训练模型
    mlp = MLPClassifier(**mlp_config)
    mlp.fit(X_train, y_train, sample_weight=sample_weight)

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


def plot_training_curves(mlp, save_dir='figures'):
    """绘制训练曲线（修复版）"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    if hasattr(mlp, 'loss_curve_') and mlp.loss_curve_ and len(mlp.loss_curve_) > 0:
        axes[0].plot(mlp.loss_curve_, 'b-', linewidth=2)
        axes[0].set_xlabel('迭代次数', fontsize=12)
        axes[0].set_ylabel('损失值', fontsize=12)
        axes[0].set_title('MLP训练损失曲线', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # 标记最终损失
        final_loss = mlp.loss_curve_[-1]
        axes[0].axhline(y=final_loss, color='red', linestyle='--', alpha=0.7)
        axes[0].text(len(mlp.loss_curve_) * 0.7, final_loss * 1.05,
                     f'最终损失: {final_loss:.4f}', fontsize=9, color='red')
    else:
        axes[0].text(0.5, 0.5, '无损失曲线数据', ha='center', va='center', fontsize=12)
        axes[0].set_title('MLP训练损失曲线', fontsize=12, fontweight='bold')

    # 验证分数曲线
    if hasattr(mlp, 'validation_scores_') and mlp.validation_scores_ and len(mlp.validation_scores_) > 0:
        axes[1].plot(mlp.validation_scores_, 'g-', linewidth=2)
        axes[1].set_xlabel('迭代次数', fontsize=12)
        axes[1].set_ylabel('验证分数', fontsize=12)
        axes[1].set_title('MLP验证分数曲线', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # 标记最佳分数
        best_score = max(mlp.validation_scores_)
        best_iter = np.argmax(mlp.validation_scores_)
        axes[1].plot(best_iter, best_score, 'ro', markersize=8)
        axes[1].annotate(f'最佳: {best_score:.4f}',
                         xy=(best_iter, best_score),
                         xytext=(best_iter + 3, best_score - 0.03),
                         fontsize=10,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    else:
        axes[1].text(0.5, 0.5, '无验证分数数据', ha='center', va='center', fontsize=12)
        axes[1].set_title('MLP验证分数曲线', fontsize=12, fontweight='bold')

    plt.suptitle(f'训练过程分析 (迭代次数: {mlp.n_iter_})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 训练曲线已保存: {save_dir}/training_curves.png")


def plot_confusion_matrix_custom(y_test, y_pred, le, save_dir='figures'):
    """绘制混淆矩阵"""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, cmap='Blues', values_format='d')

    ax.set_title('MLP预测混淆矩阵 (Lasso特征选择后)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存: {save_dir}/confusion_matrix.png")


def plot_performance_metrics(y_test, y_pred, le, save_dir='figures'):
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
    ax.set_title('各类别性能指标', fontsize=14, fontweight='bold')
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


def plot_feature_importance_mlp(mlp, X_train_selected, save_dir='figures'):
    """绘制MLP特征重要性（基于权重绝对值）"""
    os.makedirs(save_dir, exist_ok=True)

    # 使用第一层权重的绝对值作为特征重要性
    first_layer_weights = np.abs(mlp.coefs_[0])
    feature_importance = np.mean(first_layer_weights, axis=1)

    # 显示前20个最重要的特征
    top_n = min(20, len(feature_importance))
    top_idx = np.argsort(feature_importance)[-top_n:]
    top_importance = feature_importance[top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))
    bars = ax.barh(range(top_n), top_importance, color=colors)

    ax.set_xlabel('平均权重绝对值', fontsize=12)
    ax.set_ylabel('特征索引', fontsize=12)
    ax.set_title(f'MLP特征重要性 (Top {top_n}，Lasso选择后)', fontsize=14, fontweight='bold')

    # 添加数值标签
    for i, (bar, imp) in enumerate(zip(bars, top_importance)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{imp:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mlp_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ MLP特征重要性图已保存: {save_dir}/mlp_feature_importance.png")


def print_detailed_report(y_test, y_pred, le):
    """打印详细的分类报告"""
    print("\n" + "=" * 60)
    print("详细分类报告")
    print("=" * 60)

    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print(report)

    # 计算并打印各类别准确率
    from sklearn.metrics import accuracy_score
    for i, class_name in enumerate(le.classes_):
        class_mask = (y_test == i)
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
            print(f"{class_name} 准确率: {class_acc:.4f}")


def main():
    """主函数"""
    print("=" * 60)
    print("Lasso特征选择 + MLP 完整流程")
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

    # 3. Lasso特征选择
    X_train_selected, X_test_selected, selector, lasso = lasso_feature_selection(
        X_train, y_train, X_test
    )

    # 4. 可视化Lasso结果
    print("\n" + "=" * 60)
    print("生成可视化图表")
    print("=" * 60)

    # 创建保存目录
    save_dir = 'figures_lasso_mlp'
    os.makedirs(save_dir, exist_ok=True)

    # Lasso相关图表
    plot_lasso_coefficients(lasso, None, save_dir)
    plot_feature_selection_comparison(X_train.shape[1], X_train_selected.shape[1], save_dir)

    # 5. 训练MLP
    mlp, y_pred, f1 = train_mlp_with_lasso(
        X_train_selected, y_train, X_test_selected, y_test, save_dir
    )

    # 6. MLP可视化
    plot_training_curves(mlp, save_dir)
    plot_confusion_matrix_custom(y_test, y_pred, le, save_dir)
    plot_performance_metrics(y_test, y_pred, le, save_dir)
    plot_feature_importance_mlp(mlp, X_train_selected, save_dir)

    # 7. 打印详细报告
    print_detailed_report(y_test, y_pred, le)

    # 8. 总结
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)
    print(f"原始特征数: {X.shape[1]}")
    print(f"Lasso选择后特征数: {X_train_selected.shape[1]}")
    print(f"特征压缩率: {(1 - X_train_selected.shape[1] / X.shape[1]) * 100:.1f}%")
    print(f"MLP结构: {mlp.hidden_layer_sizes}")
    print(f"最终测试集F1分数: {f1:.4f}")
    print(f"\n生成的图表保存在: {save_dir}/")
    print("  - lasso_coefficients.png (Lasso系数)")
    print("  - feature_selection_comparison.png (特征选择对比)")
    print("  - training_curves.png (训练曲线)")
    print("  - confusion_matrix.png (混淆矩阵)")
    print("  - performance_metrics.png (性能指标)")
    print("  - mlp_feature_importance.png (MLP特征重要性)")


if __name__ == "__main__":
    main()