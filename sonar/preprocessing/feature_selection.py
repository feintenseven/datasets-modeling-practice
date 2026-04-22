#!/usr/bin/env python3
"""
特征选择模块（LASSO）
功能：使用Logistic Regression + L1正则进行特征选择
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

class LassoFeatureSelector:
    """LASSO特征选择器"""

    def __init__(self, C=1.0, random_state=42):
        """
        初始化LASSO特征选择器

        Parameters:
        -----------
        C : float, default=1.0
            正则化强度的倒数，较小的值表示更强的正则化
        random_state : int, default=42
            随机种子
        """
        self.C = C
        self.random_state = random_state
        self.lasso = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=C,
            random_state=random_state,
            max_iter=1000
        )
        self.selected_indices = None
        self.selected_features = None
        self.coefficients = None
        self.n_features_original = None
        self.n_features_selected = None

    def fit(self, X, y):
        """
        拟合LASSO模型并选择特征

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            训练数据
        y : array-like of shape (n_samples,)
            目标值

        Returns:
        --------
        self : object
            返回实例本身
        """
        print("正在使用LASSO进行特征选择...")

        self.n_features_original = X.shape[1]

        # 拟合LASSO模型
        self.lasso.fit(X, y)

        # 获取系数
        self.coefficients = self.lasso.coef_[0]

        # 选择非零系数的特征
        self.selected_indices = np.where(self.coefficients != 0)[0]
        self.n_features_selected = len(self.selected_indices)

        print(f"特征选择完成:")
        print(f"  原始特征数量: {self.n_features_original}")
        print(f"  选择后特征数量: {self.n_features_selected}")
        print(f"  特征减少比例: {(1 - self.n_features_selected/self.n_features_original)*100:.1f}%")

        return self

    def transform(self, X):
        """
        使用选择的特征转换数据

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            要转换的数据

        Returns:
        --------
        X_selected : array-like of shape (n_samples, n_selected_features)
            转换后的数据
        """
        if self.selected_indices is None:
            raise ValueError("必须先调用fit方法")

        return X[:, self.selected_indices]

    def fit_transform(self, X, y):
        """
        拟合并转换数据

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            训练数据
        y : array-like of shape (n_samples,)
            目标值

        Returns:
        --------
        X_selected : array-like of shape (n_samples, n_selected_features)
            转换后的数据
        """
        self.fit(X, y)
        return self.transform(X)

    def get_selected_features(self, feature_names=None):
        """
        获取选择的特征名称

        Parameters:
        -----------
        feature_names : list, optional
            原始特征名称列表

        Returns:
        --------
        selected_features : list
            选择的特征名称
        """
        if self.selected_indices is None:
            raise ValueError("必须先调用fit方法")

        if feature_names is None:
            self.selected_features = [f"feature_{i}" for i in self.selected_indices]
        else:
            self.selected_features = [feature_names[i] for i in self.selected_indices]

        return self.selected_features

    def get_feature_importance(self, feature_names=None):
        """
        获取特征重要性（系数的绝对值）

        Parameters:
        -----------
        feature_names : list, optional
            特征名称列表

        Returns:
        --------
        importance_df : pandas.DataFrame
            特征重要性DataFrame
        """
        if self.coefficients is None:
            raise ValueError("必须先调用fit方法")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.coefficients))]

        importance = np.abs(self.coefficients)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.coefficients,
            'importance': importance,
            'selected': [i in self.selected_indices for i in range(len(self.coefficients))]
        }).sort_values('importance', ascending=False)

        return importance_df

    def save_selected_features(self, path=None):
        """
        保存选择的特征信息

        Parameters:
        -----------
        path : str, optional
            保存路径，如果为None则使用默认路径
        """
        if self.selected_indices is None:
            raise ValueError("必须先调用fit方法")

        if path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(project_root, 'outputs', 'selected_features.json')

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 准备保存的数据
        save_data = {
            'selected_indices': self.selected_indices.tolist(),
            'selected_features': self.selected_features if self.selected_features else [],
            'n_features_original': self.n_features_original,
            'n_features_selected': self.n_features_selected,
            'coefficients': self.coefficients.tolist() if self.coefficients is not None else []
        }

        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"选择的特征信息已保存到: {path}")

    def load_selected_features(self, path=None):
        """
        加载选择的特征信息

        Parameters:
        -----------
        path : str, optional
            加载路径，如果为None则使用默认路径
        """
        if path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(project_root, 'outputs', 'selected_features.json')

        with open(path, 'r') as f:
            save_data = json.load(f)

        self.selected_indices = np.array(save_data['selected_indices'])
        self.selected_features = save_data['selected_features']
        self.n_features_original = save_data['n_features_original']
        self.n_features_selected = save_data['n_features_selected']
        self.coefficients = np.array(save_data['coefficients']) if save_data['coefficients'] else None

        print(f"选择的特征信息已从 {path} 加载")

    def plot_feature_importance(self, feature_names=None, top_n=20, save_path=None):
        """
        绘制特征重要性图

        Parameters:
        -----------
        feature_names : list, optional
            特征名称列表
        top_n : int, default=20
            显示前N个最重要的特征
        save_path : str, optional
            保存路径，如果为None则不保存
        """
        if self.coefficients is None:
            raise ValueError("必须先调用fit方法")

        importance_df = self.get_feature_importance(feature_names)

        # 选择前top_n个特征
        plot_df = importance_df.head(top_n).copy()
        plot_df['color'] = plot_df['selected'].map({True: 'green', False: 'red'})

        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(plot_df)), plot_df['importance'], color=plot_df['color'])
        plt.yticks(range(len(plot_df)), plot_df['feature'])
        plt.xlabel('特征重要性（系数绝对值）')
        plt.title(f'LASSO特征重要性（前{top_n}个特征）')
        plt.gca().invert_yaxis()

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='被选择的特征'),
            Patch(facecolor='red', label='未被选择的特征')
        ]
        plt.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征重要性图已保存到: {save_path}")

        plt.show()

def main():
    """主函数：测试特征选择模块"""
    print("=" * 50)
    print("LASSO特征选择模块测试")
    print("=" * 50)

    # 导入数据预处理模块
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from preprocessing.scaler import DataPreprocessor

    # 加载和预处理数据
    preprocessor = DataPreprocessor()
    X, y, feature_names = preprocessor.load_data()
    X_scaled = preprocessor.standardize()

    # 创建特征选择器
    selector = LassoFeatureSelector(C=0.1)  # 使用较强的正则化

    # 进行特征选择
    X_selected = selector.fit_transform(X_scaled, y)

    # 获取选择的特征
    selected_features = selector.get_selected_features(feature_names)

    print(f"\n选择的特征 ({selector.n_features_selected}个):")
    for i, feature in enumerate(selected_features[:20], 1):
        print(f"  {i:2d}. {feature}")
    if len(selected_features) > 20:
        print(f"  ... 还有{len(selected_features)-20}个特征")

    # 获取特征重要性
    importance_df = selector.get_feature_importance(feature_names)
    print(f"\n最重要的10个特征:")
    print(importance_df[['feature', 'coefficient', 'selected']].head(10).to_string())

    # 保存选择的特征
    selector.save_selected_features()

    # 绘制特征重要性图
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(project_root, 'outputs', 'feature_importance.png')
    selector.plot_feature_importance(feature_names, top_n=20, save_path=save_path)

    print("\nLASSO特征选择模块测试完成！")

if __name__ == "__main__":
    main()