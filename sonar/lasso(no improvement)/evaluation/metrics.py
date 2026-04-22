#!/usr/bin/env python3
"""
评估模块
功能：模型评估、性能比较、可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
import json
import os
from datetime import datetime

class ModelEvaluator:
    """模型评估器"""

    def __init__(self, model=None, model_name="Model"):
        """
        初始化模型评估器

        Parameters:
        -----------
        model : object, optional
            要评估的模型，必须有predict和predict_proba方法
        model_name : str, default="Model"
            模型名称
        """
        self.model = model
        self.model_name = model_name
        self.metrics = {}
        self.predictions = {}
        self.feature_names = None

    def evaluate(self, X_test, y_test, X_train=None, y_train=None):
        """
        评估模型性能

        Parameters:
        -----------
        X_test : array-like of shape (n_samples, n_features)
            测试特征
        y_test : array-like of shape (n_samples,)
            测试标签
        X_train : array-like, optional
            训练特征（用于计算训练集性能）
        y_train : array-like, optional
            训练标签

        Returns:
        --------
        dict: 评估指标
        """
        print(f"正在评估模型: {self.model_name}")

        # 测试集预测
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None

        # 计算基本指标
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        # 如果有概率预测，计算ROC和PR曲线
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)

            self.metrics.update({
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'precision_curve': precision.tolist(),
                'recall_curve': recall.tolist()
            })

        # 训练集性能（如果提供）
        if X_train is not None and y_train is not None:
            y_train_pred = self.model.predict(X_train)
            self.metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
            self.metrics['train_precision'] = precision_score(y_train, y_train_pred, average='weighted')
            self.metrics['train_recall'] = recall_score(y_train, y_train_pred, average='weighted')
            self.metrics['train_f1'] = f1_score(y_train, y_train_pred, average='weighted')

        # 保存预测结果
        self.predictions = {
            'y_test': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
            'y_pred': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
            'y_pred_proba': y_pred_proba.tolist() if y_pred_proba is not None and hasattr(y_pred_proba, 'tolist') else None
        }

        self._print_metrics()
        return self.metrics

    def _print_metrics(self):
        """打印评估指标"""
        print(f"\n{self.model_name} 评估结果:")
        print("=" * 40)
        print(f"准确率 (Accuracy):  {self.metrics['accuracy']:.4f}")
        print(f"精确率 (Precision): {self.metrics['precision']:.4f}")
        print(f"召回率 (Recall):    {self.metrics['recall']:.4f}")
        print(f"F1分数 (F1-score):  {self.metrics['f1']:.4f}")

        if 'roc_auc' in self.metrics:
            print(f"ROC AUC:           {self.metrics['roc_auc']:.4f}")
            print(f"PR AUC:            {self.metrics['pr_auc']:.4f}")

        if 'train_accuracy' in self.metrics:
            print(f"\n训练集准确率:      {self.metrics['train_accuracy']:.4f}")
            print(f"测试集准确率:      {self.metrics['accuracy']:.4f}")
            print(f"过拟合程度:        {self.metrics['train_accuracy'] - self.metrics['accuracy']:.4f}")

        print("\n分类报告:")
        report_df = pd.DataFrame(self.metrics['classification_report']).transpose()
        print(report_df.to_string())

    def plot_confusion_matrix(self, save_path=None):
        """
        绘制混淆矩阵

        Parameters:
        -----------
        save_path : str, optional
            保存路径，如果为None则不保存
        """
        cm = np.array(self.metrics['confusion_matrix'])

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Rock (0)', 'Mine (1)'],
                    yticklabels=['Rock (0)', 'Mine (1)'])
        plt.title(f'{self.model_name} - 混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵图已保存到: {save_path}")

        plt.show()

    def plot_roc_curve(self, save_path=None):
        """
        绘制ROC曲线

        Parameters:
        -----------
        save_path : str, optional
            保存路径，如果为None则不保存
        """
        if 'roc_auc' not in self.metrics:
            print("没有ROC曲线数据")
            return

        fpr = self.metrics['fpr']
        tpr = self.metrics['tpr']
        roc_auc = self.metrics['roc_auc']

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC曲线 (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (False Positive Rate)')
        plt.ylabel('真正率 (True Positive Rate)')
        plt.title(f'{self.model_name} - ROC曲线')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC曲线图已保存到: {save_path}")

        plt.show()

    def plot_precision_recall_curve(self, save_path=None):
        """
        绘制精确率-召回率曲线

        Parameters:
        -----------
        save_path : str, optional
            保存路径，如果为None则不保存
        """
        if 'pr_auc' not in self.metrics:
            print("没有精确率-召回率曲线数据")
            return

        precision = self.metrics['precision_curve']
        recall = self.metrics['recall_curve']
        pr_auc = self.metrics['pr_auc']

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'PR曲线 (AUC = {pr_auc:.3f})')
        plt.xlabel('召回率 (Recall)')
        plt.ylabel('精确率 (Precision)')
        plt.title(f'{self.model_name} - 精确率-召回率曲线')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR曲线图已保存到: {save_path}")

        plt.show()

    def save_results(self, path=None):
        """
        保存评估结果

        Parameters:
        -----------
        path : str, optional
            保存路径，如果为None则使用默认路径
        """
        if path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(project_root, 'outputs', f'evaluation_{self.model_name}_{timestamp}.json')

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 准备保存的数据
        save_data = {
            'model_name': self.model_name,
            'evaluation_date': datetime.now().isoformat(),
            'metrics': self.metrics,
            'predictions': self.predictions
        }

        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"评估结果已保存到: {path}")
        return path

class ModelComparator:
    """模型比较器"""

    def __init__(self):
        """初始化模型比较器"""
        self.models = {}
        self.results = {}

    def add_model(self, model, model_name, evaluator=None):
        """
        添加模型到比较器

        Parameters:
        -----------
        model : object
            要比较的模型
        model_name : str
            模型名称
        evaluator : ModelEvaluator, optional
            如果已经评估过，可以直接添加评估器
        """
        if evaluator is None:
            evaluator = ModelEvaluator(model, model_name)
        self.models[model_name] = evaluator

    def compare_models(self, X_test, y_test, X_train=None, y_train=None):
        """
        比较所有模型的性能

        Parameters:
        -----------
        X_test : array-like
            测试特征
        y_test : array-like
            测试标签
        X_train : array-like, optional
            训练特征
        y_train : array-like, optional
            训练标签

        Returns:
        --------
        pandas.DataFrame: 比较结果
        """
        print("正在比较模型性能...")
        print("=" * 60)

        comparison_results = []

        for model_name, evaluator in self.models.items():
            print(f"\n评估模型: {model_name}")
            print("-" * 40)

            # 评估模型
            metrics = evaluator.evaluate(X_test, y_test, X_train, y_train)
            self.results[model_name] = metrics

            # 收集结果
            result = {
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1']
            }

            if 'roc_auc' in metrics:
                result['ROC AUC'] = metrics['roc_auc']
                result['PR AUC'] = metrics['pr_auc']

            if 'train_accuracy' in metrics:
                result['Train Accuracy'] = metrics['train_accuracy']
                result['Overfitting'] = metrics['train_accuracy'] - metrics['accuracy']

            comparison_results.append(result)

        # 创建比较表格
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

        print("\n" + "=" * 60)
        print("模型性能比较:")
        print("=" * 60)
        print(comparison_df.to_string(index=False))

        return comparison_df

    def plot_comparison(self, metric='Accuracy', save_path=None):
        """
        绘制模型比较图

        Parameters:
        -----------
        metric : str, default='Accuracy'
            要比较的指标
        save_path : str, optional
            保存路径，如果为None则不保存
        """
        if not self.results:
            print("没有评估结果可比较")
            return

        models = list(self.results.keys())
        values = [self.results[model][metric.lower()] for model in models]

        # 按值排序
        sorted_indices = np.argsort(values)[::-1]
        models = [models[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(models)), values, color='skyblue', edgecolor='black')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        plt.ylabel(metric)
        plt.title(f'模型性能比较 - {metric}')
        plt.grid(True, alpha=0.3, axis='y')

        # 在柱子上添加数值
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"模型比较图已保存到: {save_path}")

        plt.show()

    def save_comparison_results(self, path=None):
        """
        保存比较结果

        Parameters:
        -----------
        path : str, optional
            保存路径，如果为None则使用默认路径
        """
        if path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(project_root, 'outputs', f'model_comparison_{timestamp}.json')

        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_data = {
            'comparison_date': datetime.now().isoformat(),
            'results': self.results
        }

        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"模型比较结果已保存到: {path}")
        return path

def main():
    """主函数：测试评估模块"""
    print("=" * 50)
    print("评估模块测试")
    print("=" * 50)

    # 导入必要的模块
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from preprocessing.scaler import DataPreprocessor
    from preprocessing.feature_selection import LassoFeatureSelector
    from models.mlp_model import MLPModel

    # 加载和预处理数据
    print("1. 加载和预处理数据...")
    preprocessor = DataPreprocessor()
    X, y, feature_names = preprocessor.load_data()
    X_scaled = preprocessor.standardize()

    # 特征选择
    print("\n2. 特征选择...")
    selector = LassoFeatureSelector(C=0.1)
    X_selected = selector.fit_transform(X_scaled, y)

    # 分割数据
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    # 创建和训练MLP模型
    print("\n3. 创建和训练MLP模型...")
    mlp = MLPModel(hidden_layer_sizes=(32, 16))
    mlp.fit(X_train, y_train)

    # 评估单个模型
    print("\n4. 评估单个模型...")
    evaluator = ModelEvaluator(mlp.model, "MLP with LASSO")
    metrics = evaluator.evaluate(X_test, y_test, X_train, y_train)

    # 绘制评估图表
    print("\n5. 绘制评估图表...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 混淆矩阵
    cm_path = os.path.join(project_root, 'outputs', 'confusion_matrix.png')
    evaluator.plot_confusion_matrix(save_path=cm_path)

    # ROC曲线
    roc_path = os.path.join(project_root, 'outputs', 'roc_curve.png')
    evaluator.plot_roc_curve(save_path=roc_path)

    # PR曲线
    pr_path = os.path.join(project_root, 'outputs', 'pr_curve.png')
    evaluator.plot_precision_recall_curve(save_path=pr_path)

    # 保存评估结果
    print("\n6. 保存评估结果...")
    evaluator.save_results()

    # 模型比较示例（创建另一个模型进行比较）
    print("\n7. 模型比较示例...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    # 创建比较模型
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)

    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)

    # 创建比较器
    comparator = ModelComparator()
    comparator.add_model(mlp.model, "MLP with LASSO", evaluator)
    comparator.add_model(lr_model, "Logistic Regression")
    comparator.add_model(svm_model, "SVM")

    # 比较模型
    comparison_df = comparator.compare_models(X_test, y_test, X_train, y_train)

    # 绘制比较图
    comparison_path = os.path.join(project_root, 'outputs', 'model_comparison.png')
    comparator.plot_comparison(metric='Accuracy', save_path=comparison_path)

    # 保存比较结果
    comparator.save_comparison_results()

    print("\n评估模块测试完成！")

if __name__ == "__main__":
    main()