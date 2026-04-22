#!/usr/bin/env python3
"""
MLP模型模块
功能：构建、训练和保存MLP分类器
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class MLPModel:
    """MLP分类器模型"""

    def __init__(self, hidden_layer_sizes=(24, 12), random_state=42):
        """
        初始化MLP模型

        Parameters:
        -----------
        hidden_layer_sizes : tuple, default=(24, 12)
            隐藏层大小，例如(24, 12)表示两个隐藏层，分别有24和12个神经元
        random_state : int, default=42
            随机种子
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state

        # 创建MLP分类器
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=0.005,  # L2正则化参数，从0.001增加到0.005以适度增强正则化
            batch_size=16,
            learning_rate_init=0.001,
            max_iter=200,  # 适度减少最大迭代次数
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=8,  # 适度减少n_iter_no_change以更早停止
            random_state=random_state,
            verbose=False
        )

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.training_history = None
        self.cv_scores = None
        self.feature_names = None

    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        """
        分割训练集和测试集

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            特征数据
        y : array-like of shape (n_samples,)
            目标值
        test_size : float, default=0.2
            测试集比例
        random_state : int, default=42
            随机种子

        Returns:
        --------
        tuple: (X_train, X_test, y_train, y_test)
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"数据分割完成:")
        print(f"  训练集大小: {self.X_train.shape[0]}")
        print(f"  测试集大小: {self.X_test.shape[0]}")
        print(f"  特征数量: {self.X_train.shape[1]}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def fit(self, X_train=None, y_train=None):
        """
        训练MLP模型

        Parameters:
        -----------
        X_train : array-like, optional
            训练特征，如果为None则使用self.X_train
        y_train : array-like, optional
            训练标签，如果为None则使用self.y_train

        Returns:
        --------
        self : object
            返回实例本身
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train

        if X_train is None or y_train is None:
            raise ValueError("必须先提供训练数据或调用train_test_split")

        print("正在训练MLP模型...")
        print(f"  网络结构: {self.hidden_layer_sizes}")
        print(f"  正则化参数 (alpha): {self.model.alpha}")
        print(f"  批次大小: {self.model.batch_size}")
        print(f"  学习率: {self.model.learning_rate_init}")

        # 训练模型
        self.model.fit(X_train, y_train)

        # 获取训练历史
        self.training_history = {
            'loss_curve': self.model.loss_curve_ if hasattr(self.model, 'loss_curve_') else None,
            'validation_scores': self.model.validation_scores_ if hasattr(self.model, 'validation_scores_') else None,
            'best_loss': self.model.best_loss_ if hasattr(self.model, 'best_loss_') else None,
            'n_iter': self.model.n_iter_,
            'n_layers': self.model.n_layers_,
            'n_outputs': self.model.n_outputs_
        }

        print(f"训练完成:")
        print(f"  迭代次数: {self.model.n_iter_}")
        if hasattr(self.model, 'loss_') and self.model.loss_ is not None:
            print(f"  最终损失: {self.model.loss_:.4f}")
        if hasattr(self.model, 'best_loss_') and self.model.best_loss_ is not None:
            print(f"  最佳损失: {self.model.best_loss_:.4f}")

        return self

    def predict(self, X=None):
        """
        使用训练好的模型进行预测

        Parameters:
        -----------
        X : array-like, optional
            要预测的数据，如果为None则使用self.X_test

        Returns:
        --------
        y_pred : array-like
            预测结果
        """
        if X is None:
            X = self.X_test

        if self.model is None:
            raise ValueError("必须先训练模型")

        return self.model.predict(X)

    def predict_proba(self, X=None):
        """
        预测概率

        Parameters:
        -----------
        X : array-like, optional
            要预测的数据，如果为None则使用self.X_test

        Returns:
        --------
        proba : array-like of shape (n_samples, n_classes)
            每个类别的概率
        """
        if X is None:
            X = self.X_test

        if self.model is None:
            raise ValueError("必须先训练模型")

        return self.model.predict_proba(X)

    def evaluate(self, X=None, y=None):
        """
        评估模型性能

        Parameters:
        -----------
        X : array-like, optional
            评估数据，如果为None则使用self.X_test
        y : array-like, optional
            真实标签，如果为None则使用self.y_test

        Returns:
        --------
        dict: 评估指标
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test

        if self.model is None:
            raise ValueError("必须先训练模型")

        y_pred = self.predict(X)

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y, y_pred)
        }

        return metrics

    def cross_validate(self, X, y, cv=5):
        """
        交叉验证

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            特征数据
        y : array-like of shape (n_samples,)
            目标值
        cv : int, default=5
            交叉验证折数

        Returns:
        --------
        dict: 交叉验证结果
        """
        print(f"正在进行{cv}折交叉验证...")

        # 计算交叉验证分数
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        self.cv_scores = scores

        cv_results = {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores)
        }

        print(f"交叉验证结果:")
        print(f"  各折准确率: {scores}")
        print(f"  平均准确率: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        print(f"  范围: {cv_results['min_score']:.4f} - {cv_results['max_score']:.4f}")

        return cv_results

    def save_model(self, path=None):
        """
        保存模型

        Parameters:
        -----------
        path : str, optional
            保存路径，如果为None则使用默认路径
        """
        if self.model is None:
            raise ValueError("必须先训练模型")

        if path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(project_root, 'outputs', f'mlp_model_{timestamp}.pkl')

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

        # 同时保存模型配置
        config = {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'random_state': self.random_state,
            'model_params': self.model.get_params(),
            'training_history': self.training_history
        }
        config_path = path.replace('.pkl', '_config.json')

        import json
        # 将numpy数组转换为列表以便JSON序列化
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj

        with open(config_path, 'w') as f:
            json.dump(config, f, default=convert_for_json, indent=2)

        print(f"模型已保存到: {path}")
        print(f"模型配置已保存到: {config_path}")

        return path

    def load_model(self, path):
        """
        加载模型

        Parameters:
        -----------
        path : str
            模型文件路径
        """
        self.model = joblib.load(path)
        print(f"模型已从 {path} 加载")

    def plot_training_history(self, save_path=None):
        """
        绘制训练历史

        Parameters:
        -----------
        save_path : str, optional
            保存路径，如果为None则不保存
        """
        if self.training_history is None or self.training_history['loss_curve'] is None:
            print("没有训练历史数据可绘制")
            return

        loss_curve = self.training_history['loss_curve']
        validation_scores = self.training_history['validation_scores']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 绘制损失曲线
        axes[0].plot(loss_curve, label='Training Loss', linewidth=2)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 如果有验证分数，绘制验证分数曲线
        if validation_scores is not None:
            axes[1].plot(validation_scores, label='Validation Score', color='orange', linewidth=2)
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Validation Accuracy Curve')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存到: {save_path}")

        plt.show()

    def get_model_summary(self):
        """
        获取模型摘要

        Returns:
        --------
        dict: 模型摘要信息
        """
        if self.model is None:
            raise ValueError("必须先训练模型")

        summary = {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'n_layers': self.model.n_layers_,
            'n_outputs': self.model.n_outputs_,
            'n_iter': self.model.n_iter_,
            'final_loss': float(self.model.loss_) if hasattr(self.model, 'loss_') else None,
            'activation': self.model.activation,
            'solver': self.model.solver,
            'alpha': self.model.alpha,
            'batch_size': self.model.batch_size,
            'learning_rate_init': self.model.learning_rate_init
        }

        if self.training_history:
            summary.update({
                'best_loss': float(self.training_history['best_loss']) if self.training_history['best_loss'] else None,
                'has_validation_scores': self.training_history['validation_scores'] is not None
            })

        return summary

def main():
    """主函数：测试MLP模型模块"""
    print("=" * 50)
    print("MLP模型模块测试")
    print("=" * 50)

    # 导入必要的模块
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from preprocessing.scaler import DataPreprocessor
    from preprocessing.feature_selection import LassoFeatureSelector

    # 加载和预处理数据
    print("1. 加载和预处理数据...")
    preprocessor = DataPreprocessor()
    X, y, feature_names = preprocessor.load_data()
    X_scaled = preprocessor.standardize()

    # 特征选择
    print("\n2. 特征选择...")
    selector = LassoFeatureSelector(C=0.1)
    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = selector.get_selected_features(feature_names)

    # 创建MLP模型
    print("\n3. 创建MLP模型...")
    mlp = MLPModel(hidden_layer_sizes=(24, 12))

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = mlp.train_test_split(X_selected, y, test_size=0.2)

    # 训练模型
    print("\n4. 训练模型...")
    mlp.fit(X_train, y_train)

    # 评估模型
    print("\n5. 评估模型...")
    metrics = mlp.evaluate(X_test, y_test)

    print(f"\n测试集性能:")
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  精确率: {metrics['precision']:.4f}")
    print(f"  召回率: {metrics['recall']:.4f}")
    print(f"  F1分数: {metrics['f1']:.4f}")

    # 交叉验证
    print("\n6. 交叉验证...")
    cv_results = mlp.cross_validate(X_selected, y, cv=5)

    # 获取模型摘要
    print("\n7. 模型摘要...")
    summary = mlp.get_model_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # 保存模型
    print("\n8. 保存模型...")
    model_path = mlp.save_model()

    # 绘制训练历史
    print("\n9. 绘制训练历史...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(project_root, 'outputs', 'training_history.png')
    mlp.plot_training_history(save_path=save_path)

    print("\nMLP模型模块测试完成！")

if __name__ == "__main__":
    main()