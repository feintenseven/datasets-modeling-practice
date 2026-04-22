#!/usr/bin/env python3
"""
不固定种子的MLP模型训练与评估
通过多次运行来评估模型的真实可靠性
"""

import joblib
import numpy as np
import json
import os
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing.scaler import DataPreprocessor


def train_eval_single_run(data_seed, model_seed, X_scaled, y, test_size=0.2):
    """
    单次训练和评估

    Parameters:
    - data_seed: 数据集划分的种子（None表示不固定）
    - model_seed: 模型随机种子（None表示不固定）
    - X_scaled: 特征数据
    - y: 标签
    - test_size: 测试集比例

    Returns:
    - train_acc: 训练准确率
    - test_acc: 测试准确率
    """
    # 划分数据集（如果不固定种子，传None）
    split_state = data_seed if data_seed is not None else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size,
        random_state=split_state, stratify=y
    )

    # 模型配置（如果不固定种子，random_state传None）
    model_state = model_seed if model_seed is not None else None
    best_config = {
        'hidden_layer_sizes': (24, 12),
        'alpha': 0.005,
        'batch_size': 8,
        'learning_rate_init': 0.001,
        'max_iter': 300,
        'early_stopping': True,
        'n_iter_no_change': 10,
        'validation_fraction': 0.1,
        'random_state': model_state,  # 关键：不固定种子时传None
        'activation': 'relu',
        'solver': 'adam',
        'verbose': False
    }

    model = MLPClassifier(**best_config)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'n_iter': model.n_iter_,
        'final_loss': model.loss_
    }


def evaluate_multiple_runs(n_runs=30, X_scaled=None, y=None):
    """
    多次运行评估MLP性能

    Parameters:
    - n_runs: 运行次数
    - X_scaled: 特征数据
    - y: 标签
    """
    print("\n" + "=" * 70)
    print(f"评估MLP可靠性：{n_runs}次独立运行（完全不固定种子）")
    print("=" * 70)

    results = []

    for i in range(n_runs):
        # 每次运行都使用不同的随机种子（通过不传种子实现）
        # 注意：这里每次的data_seed和model_seed都是None
        result = train_eval_single_run(
            data_seed=None,  # 不固定数据划分
            model_seed=None,  # 不固定模型初始化
            X_scaled=X_scaled,
            y=y
        )
        results.append(result)

        # 每10次打印一次进度
        if (i + 1) % 10 == 0:
            print(f"  已完成 {i + 1}/{n_runs} 次运行")

    # 统计分析
    train_accs = [r['train_acc'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    n_iters = [r['n_iter'] for r in results]

    print("\n" + "-" * 70)
    print("统计结果")
    print("-" * 70)

    print(f"\n训练准确率:")
    print(f"  平均值: {np.mean(train_accs):.4f}")
    print(f"  标准差: {np.std(train_accs):.4f}")
    print(f"  最小值: {np.min(train_accs):.4f}")
    print(f"  最大值: {np.max(train_accs):.4f}")
    print(f"  95%置信区间: [{np.percentile(train_accs, 2.5):.4f}, {np.percentile(train_accs, 97.5):.4f}]")

    print(f"\n测试准确率:")
    print(f"  平均值: {np.mean(test_accs):.4f}")
    print(f"  标准差: {np.std(test_accs):.4f}")
    print(f"  最小值: {np.min(test_accs):.4f}")
    print(f"  最大值: {np.max(test_accs):.4f}")
    print(f"  95%置信区间: [{np.percentile(test_accs, 2.5):.4f}, {np.percentile(test_accs, 97.5):.4f}]")

    print(f"\n训练-测试差距:")
    gap = np.mean(train_accs) - np.mean(test_accs)
    print(f"  平均过拟合程度: {gap:.4f}")

    print(f"\n收敛情况:")
    print(f"  平均迭代次数: {np.mean(n_iters):.1f} ± {np.std(n_iters):.1f}")

    # 判断可靠性
    print("\n" + "-" * 70)
    print("可靠性判断")
    print("-" * 70)

    test_std = np.std(test_accs)
    if test_std < 0.02:
        reliability = "✅ 高度可靠 (标准差 < 0.02)"
    elif test_std < 0.05:
        reliability = "⚠️ 基本可靠 (标准差 < 0.05)"
    else:
        reliability = "❌ 不可靠 (标准差过大)"

    print(f"  标准差: {test_std:.4f}")
    print(f"  结论: {reliability}")

    # 如果标准差大，说明需要更大的数据集
    if test_std >= 0.05:
        print(f"\n  💡 建议: 标准差较大，说明模型对随机性敏感")
        print(f"     可能需要更大的数据集或更强的正则化")

    return results


def cross_validation_evaluation(X_scaled, y, n_splits=5, n_repeats=10):
    """
    使用重复交叉验证评估（更严格的评估）

    Parameters:
    - X_scaled: 特征数据
    - y: 标签
    - n_splits: K折数
    - n_repeats: 重复次数
    """
    print("\n" + "=" * 70)
    print(f"重复交叉验证评估: {n_repeats}次重复 x {n_splits}折")
    print("=" * 70)

    best_config = {
        'hidden_layer_sizes': (24, 12),
        'alpha': 0.005,
        'batch_size': 8,
        'learning_rate_init': 0.001,
        'max_iter': 300,
        'early_stopping': True,
        'n_iter_no_change': 10,
        'validation_fraction': 0.1,
        'random_state': None,  # 不固定种子
        'activation': 'relu',
        'solver': 'adam',
        'verbose': False
    }

    all_scores = []

    for repeat in range(n_repeats):
        # 每次重复使用不同的随机划分
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)
        model = MLPClassifier(**best_config)

        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        all_scores.extend(scores)

        if (repeat + 1) % 2 == 0:
            print(f"  已完成 {repeat + 1}/{n_repeats} 次重复")

    print("\n" + "-" * 70)
    print("交叉验证统计结果")
    print("-" * 70)
    print(f"总评估次数: {len(all_scores)}")
    print(f"平均准确率: {np.mean(all_scores):.4f}")
    print(f"标准差: {np.std(all_scores):.4f}")
    print(f"95%置信区间: [{np.percentile(all_scores, 2.5):.4f}, {np.percentile(all_scores, 97.5):.4f}]")
    print(f"准确率范围: [{np.min(all_scores):.4f}, {np.max(all_scores):.4f}]")

    return all_scores


def save_best_model_fixed_seed(X_scaled, y):
    """
    保存一个固定种子的最佳模型（用于实际部署）
    注意：这个只是选一个代表性模型保存，不代表可靠性证明
    """
    print("\n" + "=" * 70)
    print("保存部署模型（固定种子，用于实际使用）")
    print("=" * 70)

    # 为了可复现的部署，这里固定种子
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    best_config = {
        'hidden_layer_sizes': (24, 12),
        'alpha': 0.005,
        'batch_size': 8,
        'learning_rate_init': 0.001,
        'max_iter': 300,
        'early_stopping': True,
        'n_iter_no_change': 10,
        'validation_fraction': 0.1,
        'random_state': 55,  # 固定种子保证可复现
        'activation': 'relu',
        'solver': 'adam',
        'verbose': False
    }

    model = MLPClassifier(**best_config)
    model.fit(X_train, y_train)

    # 保存模型
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/best_mlp_model.pkl')

    # 保存预处理器
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    preprocessor.standardize()
    joblib.dump(preprocessor.scaler, 'models/scaler.pkl')

    print(f"模型已保存到 models/best_mlp_model.pkl")
    print(f"测试准确率: {model.score(X_test, y_test):.4f}")
    print(f"\n⚠️ 注意: 这个固定种子的模型仅用于部署")
    print(f"   真正的可靠性请参考上面的多次运行结果")


def main():
    """主函数"""
    print("=" * 70)
    print("MLP可靠性评估（不固定种子版本）")
    print("=" * 70)

    # 加载数据
    print("\n1. 加载数据...")
    preprocessor = DataPreprocessor()
    X, y, feature_names = preprocessor.load_data()
    X_scaled = preprocessor.standardize()
    print(f"   数据集大小: {X_scaled.shape}")
    print(f"   类别分布: 类别0={np.sum(y == 0)}, 类别1={np.sum(y == 1)}")

    # 选择评估模式
    print("\n2. 选择评估模式:")
    print("   [1] 多次训练-测试评估 (30次)")
    print("   [2] 重复交叉验证 (10x5-fold)")
    print("   [3] 全部运行")

    choice = input("\n请选择 (1/2/3，默认3): ").strip()

    if choice == '1':
        # 多次训练-测试评估
        results = evaluate_multiple_runs(n_runs=30, X_scaled=X_scaled, y=y)

        # 可选：保存结果
        save_results = input("\n是否保存结果到文件？(y/n): ").strip().lower()
        if save_results == 'y':
            os.makedirs('evaluation_results', exist_ok=True)
            with open('evaluation_results/multi_run_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print("结果已保存到 evaluation_results/multi_run_results.json")

    elif choice == '2':
        # 重复交叉验证
        scores = cross_validation_evaluation(X_scaled, y, n_splits=5, n_repeats=10)

        # 保存结果
        os.makedirs('evaluation_results', exist_ok=True)
        with open('evaluation_results/cv_results.json', 'w') as f:
            json.dump({'scores': scores}, f, indent=2)
        print("\n结果已保存到 evaluation_results/cv_results.json")

    else:
        # 全部运行
        print("\n" + "=" * 70)
        print("运行完整评估...")
        print("=" * 70)

        # 1. 多次运行
        results = evaluate_multiple_runs(n_runs=30, X_scaled=X_scaled, y=y)

        # 2. 交叉验证
        scores = cross_validation_evaluation(X_scaled, y, n_splits=5, n_repeats=10)

        # 3. 保存结果
        os.makedirs('evaluation_results', exist_ok=True)
        with open('evaluation_results/multi_run_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        with open('evaluation_results/cv_results.json', 'w') as f:
            json.dump({'scores': scores}, f, indent=2)
        print("\n所有结果已保存到 evaluation_results/ 目录")

    # 4. 保存一个固定种子的模型用于部署
    save_deploy = input("\n是否保存一个固定种子的部署模型？(y/n): ").strip().lower()
    if save_deploy == 'y':
        save_best_model_fixed_seed(X_scaled, y)

    print("\n" + "=" * 70)
    print("评估完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()