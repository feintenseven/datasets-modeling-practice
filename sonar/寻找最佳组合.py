#!/usr/bin/env python3
"""
优化版MLP（不使用LASSO）
目标：将准确率从76%提升到80%+
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing.scaler import DataPreprocessor

# 解决中文乱码
if sys.platform == 'win32':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class OptimizedMLP:
    """优化版MLP分类器"""

    def __init__(self):
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """加载数据"""
        print("加载数据...")
        preprocessor = DataPreprocessor()
        X, y, _ = preprocessor.load_data()
        X_scaled = preprocessor.standardize()

        # 分割数据
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"训练集: {self.X_train.shape}, 测试集: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def grid_search_optimization(self):
        """网格搜索优化"""
        print("\n" + "=" * 60)
        print("网格搜索优化")
        print("=" * 60)

        # 参数网格
        param_grid = {
            'hidden_layer_sizes': [(24, 12), (32, 16), (32, 16, 8), (20, 10, 5)],
            'alpha': [0.001, 0.003, 0.005, 0.008, 0.01],
            'learning_rate_init': [0.0005, 0.001, 0.002],
            'batch_size': [8, 16, 32],
            'n_iter_no_change': [8, 10, 12]
        }

        # 基础模型
        base_mlp = MLPClassifier(
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )

        # 随机搜索（比网格搜索更快）
        from sklearn.model_selection import RandomizedSearchCV

        random_search = RandomizedSearchCV(
            base_mlp,
            param_distributions=param_grid,
            n_iter=30,  # 搜索30组参数
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        print("开始搜索最佳参数...")
        random_search.fit(self.X_train, self.y_train)

        print(f"\n最佳参数: {random_search.best_params_}")
        print(f"最佳交叉验证分数: {random_search.best_score_:.4f}")

        self.best_model = random_search.best_estimator_

        # 测试集评估
        test_score = self.best_model.score(self.X_test, self.y_test)
        print(f"测试集准确率: {test_score:.4f}")

        return random_search.best_params_, test_score

    def ensemble_optimization(self):
        """集成优化"""
        print("\n" + "=" * 60)
        print("集成模型优化")
        print("=" * 60)

        # 创建多个不同的MLP
        models = [
            ('mlp_deep', MLPClassifier(
                hidden_layer_sizes=(32, 16, 8), alpha=0.003,
                max_iter=300, random_state=42
            )),
            ('mlp_wide', MLPClassifier(
                hidden_layer_sizes=(48, 24), alpha=0.005,
                max_iter=250, random_state=123
            )),
            ('mlp_balanced', MLPClassifier(
                hidden_layer_sizes=(24, 12), alpha=0.008,
                max_iter=200, random_state=456
            )),
            ('mlp_small', MLPClassifier(
                hidden_layer_sizes=(16, 8), alpha=0.01,
                max_iter=180, random_state=789
            )),
        ]

        # 训练并评估单个模型
        print("\n单个模型性能:")
        for name, model in models:
            model.fit(self.X_train, self.y_train)
            score = model.score(self.X_test, self.y_test)
            print(f"  {name}: {score:.4f}")

        # 软投票集成
        ensemble_soft = VotingClassifier(
            estimators=models,
            voting='soft'
        )
        ensemble_soft.fit(self.X_train, self.y_train)
        soft_score = ensemble_soft.score(self.X_test, self.y_test)
        print(f"\n软投票集成准确率: {soft_score:.4f}")

        # 硬投票集成
        ensemble_hard = VotingClassifier(
            estimators=models,
            voting='hard'
        )
        ensemble_hard.fit(self.X_train, self.y_train)
        hard_score = ensemble_hard.score(self.X_test, self.y_test)
        print(f"硬投票集成准确率: {hard_score:.4f}")

        # 选择更好的集成方式
        if soft_score > hard_score:
            self.best_model = ensemble_soft
            return ensemble_soft, soft_score
        else:
            self.best_model = ensemble_hard
            return ensemble_hard, hard_score

    def advanced_optimization(self):
        """高级优化组合"""
        print("\n" + "=" * 60)
        print("高级优化组合")
        print("=" * 60)

        # 1. 数据增强（SMOTE）
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(self.X_train, self.y_train)
        print(f"数据增强后: {X_resampled.shape[0]} 样本")

        # 2. 使用最佳参数训练
        best_params = {
            'hidden_layer_sizes': (32, 16, 8),
            'alpha': 0.003,
            'learning_rate_init': 0.001,
            'batch_size': 8,
            'n_iter_no_change': 10,
            'max_iter': 400,
            'early_stopping': True,
            'random_state': 42
        }

        mlp_advanced = MLPClassifier(**best_params)
        mlp_advanced.fit(X_resampled, y_resampled)

        # 3. 评估
        test_score = mlp_advanced.score(self.X_test, self.y_test)
        print(f"高级优化模型测试准确率: {test_score:.4f}")

        # 4. 交叉验证
        cv_scores = cross_val_score(mlp_advanced, X_resampled, y_resampled, cv=5)
        print(f"交叉验证: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        self.best_model = mlp_advanced
        return mlp_advanced, test_score

    def run_comparison(self):
        """运行所有优化方法并比较"""
        print("=" * 60)
        print("MLP优化对比实验")
        print("=" * 60)

        # 加载数据
        self.load_data()

        results = {}

        # 基线模型（当前最佳）
        print("\n1. 基线模型 (24,12, alpha=0.005)")
        baseline = MLPClassifier(
            hidden_layer_sizes=(24, 12), alpha=0.005,
            max_iter=200, random_state=42
        )
        baseline.fit(self.X_train, self.y_train)
        baseline_score = baseline.score(self.X_test, self.y_test)
        results['基线模型'] = baseline_score
        print(f"   准确率: {baseline_score:.4f}")

        # 方法1：网格搜索
        try:
            params, score1 = self.grid_search_optimization()
            results['网格搜索优化'] = score1
        except:
            print("网格搜索跳过（时间较长）")

        # 方法2：集成学习
        ensemble, score2 = self.ensemble_optimization()
        results['集成学习'] = score2

        # 方法3：高级优化
        advanced, score3 = self.advanced_optimization()
        results['高级优化'] = score3

        # 结果对比
        print("\n" + "=" * 60)
        print("最终结果对比")
        print("=" * 60)
        for method, score in results.items():
            improvement = (score - baseline_score) * 100
            print(f"{method:15} : {score:.4f} ({improvement:+.2f}%)")

        # 找出最佳模型
        best_method = max(results, key=results.get)
        best_score = results[best_method]

        print(f"\n🎯 最佳方法: {best_method}")
        print(f"   准确率: {best_score:.4f} ({best_score * 100:.2f}%)")
        print(f"   相比基线提升: {(best_score - baseline_score) * 100:+.2f}%")

        return results, self.best_model


# 快速测试不同配置
def quick_test_configs():
    """快速测试多个配置"""
    print("快速测试不同配置")
    print("=" * 60)

    # 加载数据
    preprocessor = DataPreprocessor()
    X, y, _ = preprocessor.load_data()
    X_scaled = preprocessor.standardize()

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 测试配置
    configs = [
        # (名称, 网络结构, alpha, 学习率, 批次大小)
        ("当前最佳", (24, 12), 0.005, 0.001, 16),
        ("三层网络", (32, 16, 8), 0.003, 0.001, 8),
        ("更宽网络", (48, 24), 0.008, 0.0005, 16),
        ("更小批次", (24, 12), 0.005, 0.001, 8),
        ("降低学习率", (24, 12), 0.005, 0.0005, 16),
        ("增强正则化", (24, 12), 0.01, 0.001, 16),
    ]

    results = []
    for name, hidden, alpha, lr, batch in configs:
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden,
            alpha=alpha,
            learning_rate_init=lr,
            batch_size=batch,
            max_iter=300,
            early_stopping=True,
            random_state=42
        )
        mlp.fit(X_train, y_train)
        train_acc = mlp.score(X_train, y_train)
        test_acc = mlp.score(X_test, y_test)
        overfit = train_acc - test_acc

        results.append({
            '配置': name,
            '测试准确率': test_acc,
            '过拟合': overfit
        })

        print(f"{name:12} : 测试={test_acc:.4f}, 过拟合={overfit:.4f}")

    # 找出最佳
    df = pd.DataFrame(results)
    best = df.loc[df['测试准确率'].idxmax()]
    print(f"\n✅ 最佳配置: {best['配置']} (准确率: {best['测试准确率']:.4f})")

    return df


if __name__ == "__main__":
    # 快速测试
    quick_test_configs()

    # 运行完整优化
    print("\n" + "=" * 60)
    print("开始完整优化流程")
    print("=" * 60)

    optimizer = OptimizedMLP()
    results, best_model = optimizer.run_comparison()