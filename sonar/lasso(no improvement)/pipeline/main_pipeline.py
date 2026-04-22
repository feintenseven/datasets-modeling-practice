#!/usr/bin/env python3
"""
主pipeline模块
功能：整合所有模块，创建完整的工作流程
"""

import numpy as np
import pandas as pd
import os
import sys
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.scaler import DataPreprocessor
from preprocessing.feature_selection import LassoFeatureSelector
from models.mlp_model import MLPModel
from evaluation.metrics import ModelEvaluator, ModelComparator

class SonarPipeline:
    """Sonar项目主pipeline"""

    def __init__(self, data_path=None, output_dir='outputs'):
        """
        初始化pipeline

        Parameters:
        -----------
        data_path : str, optional
            数据文件路径
        output_dir : str, default='outputs'
            输出目录
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.preprocessor = None
        self.selector = None
        self.mlp_model = None
        self.evaluator = None
        self.comparator = None

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # ========== 添加以下代码解决中文乱码 ==========
        # 配置matplotlib中文字体
        import sys
        import matplotlib.pyplot as plt

        # 根据操作系统选择字体
        if sys.platform == 'win32':  # Windows
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
        elif sys.platform == 'darwin':  # macOS
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC']
        else:  # Linux
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei']

        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False
        # ========== 添加结束 ==========

        # 实验记录
        self.experiments = {}
        self.current_experiment = None

    def run_full_pipeline(self, experiment_name="full_experiment"):
        """
        运行完整的pipeline

        Parameters:
        -----------
        experiment_name : str, default="full_experiment"
            实验名称

        Returns:
        --------
        dict: 实验结果
        """
        print("=" * 60)
        print(f"开始运行完整pipeline: {experiment_name}")
        print("=" * 60)

        # 开始实验记录
        self._start_experiment(experiment_name)

        # 1. 数据预处理
        print("\n1. 数据预处理")
        print("-" * 40)
        self.preprocessor = DataPreprocessor(self.data_path)
        X, y, feature_names = self.preprocessor.load_data()
        X_scaled = self.preprocessor.standardize()

        # 2. 特征选择
        print("\n2. 特征选择 (LASSO)")
        print("-" * 40)
        self.selector = LassoFeatureSelector(C=0.1)
        X_selected = self.selector.fit_transform(X_scaled, y)
        selected_features = self.selector.get_selected_features(feature_names)

        # 保存特征选择结果
        self.selector.save_selected_features()

        # 3. 分割数据
        print("\n3. 数据分割")
        print("-" * 40)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"训练集大小: {X_train.shape[0]}")
        print(f"测试集大小: {X_test.shape[0]}")
        print(f"特征数量: {X_train.shape[1]}")

        # 4. 训练MLP模型
        print("\n4. 训练MLP模型")
        print("-" * 40)
        self.mlp_model = MLPModel(hidden_layer_sizes=(32, 16))
        self.mlp_model.train_test_split(X_selected, y)  # 保存分割信息
        self.mlp_model.fit(X_train, y_train)

        # 5. 评估MLP模型
        print("\n5. 评估MLP模型")
        print("-" * 40)
        self.evaluator = ModelEvaluator(self.mlp_model.model, "MLP with LASSO")
        mlp_metrics = self.evaluator.evaluate(X_test, y_test, X_train, y_train)

        # 6. 保存模型和结果
        print("\n6. 保存模型和结果")
        print("-" * 40)
        model_path = self.mlp_model.save_model()
        eval_path = self.evaluator.save_results()

        # 7. 绘制图表
        print("\n7. 生成可视化图表")
        print("-" * 40)
        self._generate_visualizations()

        # 8. 模型比较
        print("\n8. 模型比较")
        print("-" * 40)
        self._run_model_comparison(X_train, X_test, y_train, y_test)

        # 结束实验记录
        results = self._end_experiment({
            'model_path': model_path,
            'eval_path': eval_path,
            'mlp_metrics': mlp_metrics,
            'n_features_original': X.shape[1],
            'n_features_selected': X_selected.shape[1],
            'feature_reduction': f"{(1 - X_selected.shape[1]/X.shape[1])*100:.1f}%"
        })

        print("\n" + "=" * 60)
        print("完整pipeline运行完成！")
        print("=" * 60)

        return results

    def run_experiment_1(self):
        """
        实验1：LASSO效果对比
        比较MLP使用原始特征和使用LASSO选择特征的效果
        """
        print("=" * 60)
        print("实验1: LASSO效果对比")
        print("=" * 60)

        experiment_name = "experiment_1_lasso_effect"
        self._start_experiment(experiment_name)

        # 加载数据
        self.preprocessor = DataPreprocessor(self.data_path)
        X, y, feature_names = self.preprocessor.load_data()
        X_scaled = self.preprocessor.standardize()

        # 分割数据
        from sklearn.model_selection import train_test_split
        X_train_full, X_test_full, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # 特征选择
        self.selector = LassoFeatureSelector(C=0.1)
        X_selected = self.selector.fit_transform(X_scaled, y)
        X_train_selected, X_test_selected, _, _ = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )

        # 训练两个MLP模型
        print("\n训练MLP (原始60维特征)...")
        mlp_full = MLPModel(hidden_layer_sizes=(32, 16))
        mlp_full.fit(X_train_full, y_train)

        print("\n训练MLP (LASSO选择后特征)...")
        mlp_selected = MLPModel(hidden_layer_sizes=(32, 16))
        mlp_selected.fit(X_train_selected, y_train)

        # 评估两个模型
        evaluator_full = ModelEvaluator(mlp_full.model, "MLP (原始60维)")
        metrics_full = evaluator_full.evaluate(X_test_full, y_test, X_train_full, y_train)

        evaluator_selected = ModelEvaluator(mlp_selected.model, "MLP (LASSO后)")
        metrics_selected = evaluator_selected.evaluate(X_test_selected, y_test, X_train_selected, y_train)

        # 比较结果
        comparison_results = pd.DataFrame({
            'Model': ['MLP (原始60维)', 'MLP (LASSO后)'],
            'Accuracy': [metrics_full['accuracy'], metrics_selected['accuracy']],
            'Precision': [metrics_full['precision'], metrics_selected['precision']],
            'Recall': [metrics_full['recall'], metrics_selected['recall']],
            'F1': [metrics_full['f1'], metrics_selected['f1']],
            'ROC AUC': [metrics_full.get('roc_auc', 0), metrics_selected.get('roc_auc', 0)],
            'Features': [X_train_full.shape[1], X_train_selected.shape[1]],
            'Feature Reduction': ['0%', f"{(1 - X_train_selected.shape[1]/X_train_full.shape[1])*100:.1f}%"]
        })

        print("\n实验1结果对比:")
        print(comparison_results.to_string(index=False))

        # 绘制对比图
        self._plot_experiment_results(comparison_results, 'Accuracy',
                                     '实验1: LASSO特征选择效果对比',
                                     'lasso_effect_comparison.png')

        results = self._end_experiment({
            'comparison_results': comparison_results.to_dict('records'),
            'metrics_full': metrics_full,
            'metrics_selected': metrics_selected
        })

        return results

    def run_experiment_2(self):
        """
        实验2：网络结构对比
        比较不同隐藏层结构的MLP性能
        """
        print("=" * 60)
        print("实验2: 网络结构对比")
        print("=" * 60)

        experiment_name = "experiment_2_architecture"
        self._start_experiment(experiment_name)

        # 加载和预处理数据
        self.preprocessor = DataPreprocessor(self.data_path)
        X, y, feature_names = self.preprocessor.load_data()
        X_scaled = self.preprocessor.standardize()

        # 特征选择
        self.selector = LassoFeatureSelector(C=0.1)
        X_selected = self.selector.fit_transform(X_scaled, y)

        # 分割数据
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )

        # 定义不同的网络结构
        architectures = [
            (16,),      # 单层16个神经元
            (32, 16),   # 两层：32 -> 16
            (64, 32),   # 两层：64 -> 32
            (32, 16, 8) # 三层：32 -> 16 -> 8
        ]

        results = []

        for arch in architectures:
            print(f"\n训练MLP (结构: {arch})...")
            mlp = MLPModel(hidden_layer_sizes=arch)
            mlp.fit(X_train, y_train)

            evaluator = ModelEvaluator(mlp.model, f"MLP {arch}")
            metrics = evaluator.evaluate(X_test, y_test, X_train, y_train)

            results.append({
                'Architecture': str(arch),
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'ROC AUC': metrics.get('roc_auc', 0),
                'Layers': len(arch) + 2,  # 输入层 + 隐藏层 + 输出层
                'Neurons': sum(arch)
            })

        # 创建比较表格
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

        print("\n实验2结果对比:")
        print(comparison_df.to_string(index=False))

        # 绘制对比图
        self._plot_experiment_results(comparison_df, 'Accuracy',
                                     '实验2: 网络结构对比',
                                     'architecture_comparison.png')

        results_data = self._end_experiment({
            'comparison_results': comparison_df.to_dict('records'),
            'best_architecture': comparison_df.iloc[0]['Architecture'],
            'best_accuracy': float(comparison_df.iloc[0]['Accuracy'])
        })

        return results_data

    def run_experiment_3(self):
        """
        实验3：正则化参数对比
        比较不同alpha值对MLP性能的影响
        """
        print("=" * 60)
        print("实验3: 正则化参数对比")
        print("=" * 60)

        experiment_name = "experiment_3_regularization"
        self._start_experiment(experiment_name)

        # 加载和预处理数据
        self.preprocessor = DataPreprocessor(self.data_path)
        X, y, feature_names = self.preprocessor.load_data()
        X_scaled = self.preprocessor.standardize()

        # 特征选择
        self.selector = LassoFeatureSelector(C=0.1)
        X_selected = self.selector.fit_transform(X_scaled, y)

        # 分割数据
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )

        # 定义不同的alpha值
        alphas = [0.0001, 0.001, 0.01, 0.1]

        results = []

        for alpha in alphas:
            print(f"\n训练MLP (alpha={alpha})...")
            # 创建自定义MLP模型
            from sklearn.neural_network import MLPClassifier
            mlp = MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation='relu',
                solver='adam',
                alpha=alpha,
                batch_size=16,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                random_state=42
            )
            mlp.fit(X_train, y_train)

            evaluator = ModelEvaluator(mlp, f"MLP alpha={alpha}")
            metrics = evaluator.evaluate(X_test, y_test, X_train, y_train)

            results.append({
                'Alpha': alpha,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'ROC AUC': metrics.get('roc_auc', 0),
                'Train Accuracy': metrics.get('train_accuracy', 0),
                'Overfitting': metrics.get('train_accuracy', 0) - metrics['accuracy']
            })

        # 创建比较表格
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

        print("\n实验3结果对比:")
        print(comparison_df.to_string(index=False))

        # 绘制对比图
        self._plot_experiment_results(comparison_df, 'Accuracy',
                                     '实验3: 正则化参数对比',
                                     'regularization_comparison.png')

        # 绘制过拟合程度图
        plt.figure(figsize=(10, 6))
        plt.plot(comparison_df['Alpha'], comparison_df['Overfitting'], 'o-', linewidth=2)
        plt.xscale('log')
        plt.xlabel('Alpha (正则化强度)')
        plt.ylabel('过拟合程度 (训练准确率 - 测试准确率)')
        plt.title('正则化强度与过拟合程度关系')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'overfitting_vs_alpha.png'), dpi=300)
        plt.show()

        results_data = self._end_experiment({
            'comparison_results': comparison_df.to_dict('records'),
            'best_alpha': float(comparison_df.iloc[0]['Alpha']),
            'best_accuracy': float(comparison_df.iloc[0]['Accuracy'])
        })

        return results_data

    def _start_experiment(self, name):
        """开始实验记录"""
        self.current_experiment = {
            'name': name,
            'start_time': datetime.now().isoformat(),
            'steps': []
        }
        print(f"开始实验: {name}")

    def _end_experiment(self, results):
        """结束实验记录并保存结果"""
        if self.current_experiment:
            self.current_experiment['end_time'] = datetime.now().isoformat()
            self.current_experiment['results'] = results
            self.experiments[self.current_experiment['name']] = self.current_experiment

            # 保存实验记录
            exp_path = os.path.join(self.output_dir, f"{self.current_experiment['name']}.json")
            with open(exp_path, 'w') as f:
                json.dump(self.current_experiment, f, indent=2, default=str)

            print(f"实验记录已保存到: {exp_path}")

            exp = self.current_experiment
            self.current_experiment = None
            return exp
        return None

    def _generate_visualizations(self):
        """生成可视化图表"""
        # 特征重要性图
        if self.selector:
            feature_importance_path = os.path.join(self.output_dir, 'feature_importance.png')
            self.selector.plot_feature_importance(save_path=feature_importance_path)

        # 训练历史图
        if self.mlp_model:
            training_history_path = os.path.join(self.output_dir, 'training_history.png')
            self.mlp_model.plot_training_history(save_path=training_history_path)

        # 评估图表
        if self.evaluator:
            cm_path = os.path.join(self.output_dir, 'confusion_matrix.png')
            roc_path = os.path.join(self.output_dir, 'roc_curve.png')
            pr_path = os.path.join(self.output_dir, 'pr_curve.png')

            self.evaluator.plot_confusion_matrix(save_path=cm_path)
            self.evaluator.plot_roc_curve(save_path=roc_path)
            self.evaluator.plot_precision_recall_curve(save_path=pr_path)

    def _run_model_comparison(self, X_train, X_test, y_train, y_test):
        """运行模型比较"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier

        # 创建比较器
        self.comparator = ModelComparator()

        # 添加MLP模型
        self.comparator.add_model(self.mlp_model.model, "MLP with LASSO", self.evaluator)

        # 添加其他模型
        models = [
            (LogisticRegression(random_state=42), "Logistic Regression"),
            (SVC(probability=True, random_state=42), "SVM"),
            (DecisionTreeClassifier(random_state=42), "Decision Tree"),
            (RandomForestClassifier(random_state=42), "Random Forest")
        ]

        for model, name in models:
            print(f"\n训练{name}...")
            model.fit(X_train, y_train)
            self.comparator.add_model(model, name)

        # 比较所有模型
        comparison_df = self.comparator.compare_models(X_test, y_test, X_train, y_train)

        # 绘制比较图
        comparison_path = os.path.join(self.output_dir, 'model_comparison.png')
        self.comparator.plot_comparison(metric='Accuracy', save_path=comparison_path)

        # 保存比较结果
        self.comparator.save_comparison_results()

        return comparison_df

    def _plot_experiment_results(self, df, metric, title, filename):
        """绘制实验结果图"""
        plt.figure(figsize=(12, 6))
        x = range(len(df))
        bars = plt.bar(x, df[metric], color='lightblue', edgecolor='black')
        plt.xticks(x, df.iloc[:, 0], rotation=45, ha='right')
        plt.ylabel(metric)
        plt.title(title)
        plt.grid(True, alpha=0.3, axis='y')

        # 在柱子上添加数值
        for bar, value in zip(bars, df[metric]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"实验图表已保存到: {save_path}")

    def generate_report(self):
        """生成项目报告"""
        print("=" * 60)
        print("生成项目报告")
        print("=" * 60)

        report = {
            'project': 'Sonar数据集：LASSO + MLP分类项目',
            'generated_date': datetime.now().isoformat(),
            'experiments': self.experiments,
            'summary': self._generate_summary()
        }

        report_path = os.path.join(self.output_dir, 'project_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"项目报告已保存到: {report_path}")

        # 生成文本报告
        text_report = self._generate_text_report()
        text_report_path = os.path.join(self.output_dir, 'project_report.txt')
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write(text_report)

        print(f"文本报告已保存到: {text_report_path}")

        return report

    def _generate_summary(self):
        """生成项目摘要"""
        summary = {
            'total_experiments': len(self.experiments),
            'experiment_names': list(self.experiments.keys()),
            'best_results': {}
        }

        # 收集各个实验的最佳结果
        for exp_name, exp_data in self.experiments.items():
            if 'results' in exp_data:
                results = exp_data['results']
                if 'comparison_results' in results:
                    # 找到准确率最高的模型
                    best_acc = 0
                    best_model = None
                    for model_result in results['comparison_results']:
                        if model_result.get('Accuracy', 0) > best_acc:
                            best_acc = model_result['Accuracy']
                            best_model = model_result

                    if best_model:
                        summary['best_results'][exp_name] = {
                            'best_model': best_model.get('Model') or best_model.get('Architecture') or f"Alpha={best_model.get('Alpha')}",
                            'best_accuracy': best_acc
                        }

        return summary

    def _generate_text_report(self):
        """生成文本格式的报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("Sonar数据集：LASSO + MLP分类项目报告")
        report_lines.append("=" * 60)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"实验数量: {len(self.experiments)}")
        report_lines.append("")

        for exp_name, exp_data in self.experiments.items():
            report_lines.append(f"实验: {exp_name}")
            report_lines.append(f"开始时间: {exp_data['start_time']}")
            report_lines.append(f"结束时间: {exp_data['end_time']}")
            report_lines.append("")

            if 'results' in exp_data:
                results = exp_data['results']

                if 'comparison_results' in results:
                    report_lines.append("结果对比:")
                    # 创建表格
                    df = pd.DataFrame(results['comparison_results'])
                    report_lines.append(df.to_string(index=False))
                elif 'mlp_metrics' in results:
                    report_lines.append("MLP模型性能:")
                    metrics = results['mlp_metrics']
                    report_lines.append(f"准确率: {metrics['accuracy']:.4f}")
                    report_lines.append(f"精确率: {metrics['precision']:.4f}")
                    report_lines.append(f"召回率: {metrics['recall']:.4f}")
                    report_lines.append(f"F1分数: {metrics['f1']:.4f}")
                    if 'roc_auc' in metrics:
                        report_lines.append(f"ROC AUC: {metrics['roc_auc']:.4f}")

                report_lines.append("")

        # 添加项目摘要
        report_lines.append("项目摘要:")
        report_lines.append("-" * 40)
        summary = self._generate_summary()
        for exp_name, best_result in summary['best_results'].items():
            report_lines.append(f"{exp_name}:")
            report_lines.append(f"  最佳模型: {best_result['best_model']}")
            report_lines.append(f"  最佳准确率: {best_result['best_accuracy']:.4f}")
            report_lines.append("")

        report_lines.append("=" * 60)
        report_lines.append("报告结束")
        report_lines.append("=" * 60)

        return "\n".join(report_lines)

def main():
    """主函数：运行完整的pipeline"""
    print("Sonar数据集：LASSO + MLP分类项目")
    print("=" * 60)

    # 创建pipeline
    pipeline = SonarPipeline()

    # 运行完整pipeline
    print("\n运行完整pipeline...")
    full_results = pipeline.run_full_pipeline()

    # 运行实验1：LASSO效果对比
    print("\n\n运行实验1: LASSO效果对比...")
    exp1_results = pipeline.run_experiment_1()

    # 运行实验2：网络结构对比
    print("\n\n运行实验2: 网络结构对比...")
    exp2_results = pipeline.run_experiment_2()

    # 运行实验3：正则化参数对比
    print("\n\n运行实验3: 正则化参数对比...")
    exp3_results = pipeline.run_experiment_3()

    # 生成项目报告
    print("\n\n生成项目报告...")
    report = pipeline.generate_report()

    print("\n" + "=" * 60)
    print("所有实验完成！")
    print("=" * 60)

    # 打印关键发现
    print("\n关键发现:")
    print("-" * 40)

    summary = pipeline._generate_summary()
    for exp_name, best_result in summary['best_results'].items():
        print(f"{exp_name}:")
        print(f"  最佳模型: {best_result['best_model']}")
        print(f"  最佳准确率: {best_result['best_accuracy']:.4f}")

    print("\n项目输出文件保存在 'outputs' 目录中。")

if __name__ == "__main__":
    main()