#!/usr/bin/env python3
"""
使用学生学业表现数据集证明MLP的可靠性 - 带可视化
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
        return None, None

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
    """计算样本权重"""
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    sample_weight = np.zeros_like(y, dtype=float)
    for cls, w in zip(classes, weights):
        sample_weight[y == cls] = w
    return sample_weight


def plot_class_distribution(y, le):
    """绘制类别分布图"""
    fig, ax = plt.subplots(figsize=(8, 6))

    classes = le.classes_
    counts = np.bincount(y)
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

    bars = ax.bar(classes, counts, color=colors[:len(classes)])
    ax.set_xlabel('类别', fontsize=12)
    ax.set_ylabel('样本数量', fontsize=12)
    ax.set_title('学生学业状态类别分布', fontsize=14, fontweight='bold')

    # 在柱状图上添加数值
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(count), ha='center', va='bottom', fontsize=11)

    # 添加百分比
    total = sum(counts)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        percentage = count / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f'{percentage:.1f}%', ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ 类别分布图已保存: class_distribution.png")


def plot_training_history(history, model_name):
    """绘制训练历史（损失曲线）"""
    if not history:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    if hasattr(history, 'loss_curve_') and history.loss_curve_:
        axes[0].plot(history.loss_curve_, 'b-', linewidth=2)
        axes[0].set_xlabel('迭代次数', fontsize=12)
        axes[0].set_ylabel('损失值', fontsize=12)
        axes[0].set_title(f'{model_name} - 训练损失曲线', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

    # 验证分数曲线（如果有）
    if hasattr(history, 'validation_scores_') and history.validation_scores_:
        axes[1].plot(history.validation_scores_, 'g-', linewidth=2)
        axes[1].set_xlabel('迭代次数', fontsize=12)
        axes[1].set_ylabel('验证分数', fontsize=12)
        axes[1].set_title(f'{model_name} - 验证分数曲线', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'training_history_{model_name.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 训练曲线已保存: training_history_{model_name.replace(' ', '_')}.png")


def plot_confusion_matrix(y_test, y_pred, le, model_name):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(8, 6))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, cmap='Blues', values_format='d')

    ax.set_title(f'{model_name} - 混淆矩阵', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 混淆矩阵已保存: confusion_matrix_{model_name.replace(' ', '_')}.png")


def plot_model_comparison(results):
    """绘制模型对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    models = list(results.keys())
    means = [results[m]['mean'] for m in models]
    stds = [results[m]['std'] for m in models]

    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, means, yerr=stds, capsize=5, color=colors, edgecolor='black')

    ax.set_ylabel('加权F1分数', fontsize=12)
    ax.set_title('模型性能对比（含误差条）', fontsize=14, fontweight='bold')
    ax.set_ylim(0.6, 0.85)
    ax.grid(True, alpha=0.3, axis='y')

    # 在柱状图上添加数值
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ 模型对比图已保存: model_comparison.png")


def plot_radar_chart(results):
    """绘制雷达图（多维度对比）"""
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    models = list(results.keys())
    metrics = ['F1分数', '稳定性', '训练速度', '复杂度']

    # 为每个模型计算指标（归一化到0-1）
    f1_scores = [results[m]['mean'] for m in models]
    stabilities = [1 - results[m]['std'] * 10 for m in models]  # 标准差越小越稳定

    # 简化处理其他指标
    speeds = [0.8, 0.7, 0.9, 0.6]  # 示例值
    complexities = [0.6, 0.5, 0.8, 0.7]  # 示例值

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    for i, model in enumerate(models):
        values = [f1_scores[i], stabilities[i], speeds[i], complexities[i]]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('模型多维度对比雷达图', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('radar_chart.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ 雷达图已保存: radar_chart.png")


def plot_performance_distribution(results, model_name):
    """绘制性能分布箱线图"""
    # 这里需要保存每次运行的详细分数
    pass


def evaluate_model_stability_with_plots(model, model_name, X, y, le, n_runs=10):
    """评估模型的稳定性并生成可视化"""
    print(f"\n{'=' * 60}")
    print(f"评估: {model_name}")
    print(f"{'=' * 60}")

    test_f1_scores = []
    test_acc_scores = []
    all_predictions = []
    all_true_labels = []
    training_histories = []

    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=None
        )

        from sklearn.base import clone
        model_clone = clone(model)

        try:
            if model_name.startswith('MLP'):
                sample_weight = get_sample_weights(y_train)
                model_clone.fit(X_train, y_train, sample_weight=sample_weight)
                # 保存训练历史（仅最后一次）
                if run == n_runs - 1 and hasattr(model_clone, 'loss_curve_'):
                    training_histories.append(model_clone)
            else:
                model_clone.fit(X_train, y_train)

            y_pred = model_clone.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted')
            acc = model_clone.score(X_test, y_test)

            test_f1_scores.append(f1)
            test_acc_scores.append(acc)

            # 保存最后一次的预测结果用于混淆矩阵
            if run == n_runs - 1:
                all_predictions = y_pred
                all_true_labels = y_test

        except Exception as e:
            print(f"  运行 {run + 1} 失败: {e}")
            continue

        if (run + 1) % 5 == 0:
            print(f"  已完成 {run + 1}/{n_runs}")

    if len(test_f1_scores) == 0:
        print(f"❌ {model_name} 评估失败")
        return None

    # 统计分析
    mean_f1 = np.mean(test_f1_scores)
    std_f1 = np.std(test_f1_scores)
    mean_acc = np.mean(test_acc_scores)
    std_acc = np.std(test_acc_scores)

    print(f"\n结果 ({len(test_f1_scores)}次成功运行):")
    print(f"  平均F1分数: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  平均准确率: {mean_acc:.4f} ± {std_acc:.4f}")

    # 绘制F1分数分布
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(test_f1_scores, bins=10, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(mean_f1, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_f1:.4f}')
    ax.axvline(mean_f1 - std_f1, color='orange', linestyle=':', linewidth=2, label=f'-1σ: {mean_f1 - std_f1:.4f}')
    ax.axvline(mean_f1 + std_f1, color='orange', linestyle=':', linewidth=2, label=f'+1σ: {mean_f1 + std_f1:.4f}')
    ax.set_xlabel('加权F1分数', fontsize=12)
    ax.set_ylabel('频次', fontsize=12)
    ax.set_title(f'{model_name} - F1分数分布 ({n_runs}次运行)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'f1_distribution_{model_name.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 绘制混淆矩阵
    if len(all_predictions) > 0:
        plot_confusion_matrix(all_true_labels, all_predictions, le, model_name)

    # 绘制训练曲线（如果是MLP）
    if model_name.startswith('MLP') and training_histories:
        plot_training_history(training_histories[0], model_name)

    return {'mean': mean_f1, 'std': std_f1, 'all_scores': test_f1_scores}


def main():
    print("=" * 60)
    print("MLP可靠性评估实验（带可视化）")
    print("=" * 60)

    # 1. 加载数据
    X, y, le = load_data('dataset/data.csv')

    if X is None:
        print("\n❌ 数据加载失败，请检查文件路径和格式")
        return

    # 2. 绘制类别分布
    print("\n" + "=" * 60)
    print("数据可视化")
    print("=" * 60)
    plot_class_distribution(y, le)

    # 3. 定义模型
    models = {
        'MLP (小型)': MLPClassifier(
            hidden_layer_sizes=(16, 8),
            alpha=0.01,
            max_iter=500,
            early_stopping=True,
            random_state=None,
            verbose=False
        ),
        'MLP (中型)': MLPClassifier(
            hidden_layer_sizes=(32, 16),
            alpha=0.001,
            max_iter=500,
            early_stopping=True,
            random_state=None,
            verbose=False
        ),
        '逻辑回归': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=None
        ),
        '随机森林': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=None,
            n_jobs=-1
        )
    }

    # 4. 评估每个模型
    results = {}

    for name, model in models.items():
        result = evaluate_model_stability_with_plots(model, name, X, y, le, n_runs=10)
        if result:
            results[name] = result

    # 5. 绘制模型对比图
    print("\n" + "=" * 60)
    print("生成对比图表")
    print("=" * 60)
    plot_model_comparison(results)

    # 6. 打印总结
    print("\n" + "=" * 60)
    print("总结对比")
    print("=" * 60)
    print(f"{'模型':<20} {'平均F1':<12} {'标准差':<12}")
    print("-" * 44)
    for name, stats in results.items():
        print(f"{name:<20} {stats['mean']:.4f}       ±{stats['std']:.4f}")

    # 7. 结论
    if 'MLP (中型)' in results:
        mlp_mean = results['MLP (中型)']['mean']
        mlp_std = results['MLP (中型)']['std']

        print(f"\n{'=' * 60}")
        print("结论")
        print(f"{'=' * 60}")

        if mlp_std < 0.02:
            print(f"✅ MLP高度可靠（F1={mlp_mean:.4f}, 标准差={mlp_std:.4f} < 0.02）")
        elif mlp_std < 0.05:
            print(f"⚠️ MLP基本可靠（F1={mlp_mean:.4f}, 标准差={mlp_std:.4f} < 0.05）")
        else:
            print(f"❌ MLP不够稳定（标准差={mlp_std:.4f} > 0.05）")

        print(f"\n📊 生成的图表文件:")
        print(f"   - class_distribution.png (类别分布)")
        print(f"   - model_comparison.png (模型对比)")
        print(f"   - f1_distribution_*.png (F1分布)")
        print(f"   - confusion_matrix_*.png (混淆矩阵)")
        print(f"   - training_history_*.png (训练曲线)")


if __name__ == "__main__":
    main()