#!/usr/bin/env python3
"""
数据预处理模块
功能：读取数据、标签编码、标准化
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

class DataPreprocessor:
    """数据预处理类"""

    def __init__(self, data_path=None):
        """
        初始化数据预处理器

        Parameters:
        -----------
        data_path : str, optional
            数据文件路径，如果为None则使用默认路径
        """
        if data_path is None:
            # 使用相对于项目根目录的路径
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_path = os.path.join(project_root, 'data', 'sonar.csv')
        else:
            self.data_path = data_path

        self.scaler = StandardScaler()
        self.X = None
        self.y = None
        self.X_scaled = None
        self.feature_names = None

    def load_data(self):
        """
        加载数据

        Returns:
        --------
        tuple: (X, y, feature_names)
            X: 特征数据 (n_samples, n_features)
            y: 标签数据 (n_samples,)
            feature_names: 特征名称列表
        """
        print("正在加载数据...")

        # 读取CSV文件
        df = pd.read_csv(self.data_path)

        # 分离特征和标签
        self.X = df.iloc[:, :-1].values
        self.y = df.iloc[:, -1].values

        # 标签编码：R -> 0, M -> 1
        self.y = np.where(self.y == 'R', 0, 1)

        # 获取特征名称
        self.feature_names = df.columns[:-1].tolist()

        print(f"数据加载完成:")
        print(f"  样本数量: {self.X.shape[0]}")
        print(f"  特征数量: {self.X.shape[1]}")
        print(f"  类别分布: Rock(R)=0: {np.sum(self.y == 0)}, Mine(M)=1: {np.sum(self.y == 1)}")

        return self.X, self.y, self.feature_names

    def standardize(self, X=None, fit=True):
        """
        标准化数据

        Parameters:
        -----------
        X : array-like, optional
            要标准化的数据，如果为None则使用self.X
        fit : bool, default=True
            是否拟合scaler（训练时用True，预测时用False）

        Returns:
        --------
        array: 标准化后的数据
        """
        if X is None:
            X = self.X

        if fit:
            self.X_scaled = self.scaler.fit_transform(X)
            print("数据标准化完成（已拟合scaler）")
        else:
            self.X_scaled = self.scaler.transform(X)
            print("数据标准化完成（使用已拟合的scaler）")

        # 打印标准化后的统计信息
        print(f"  标准化后 - 均值: {self.X_scaled.mean():.4f}, 标准差: {self.X_scaled.std():.4f}")

        return self.X_scaled

    def save_scaler(self, path=None):
        """
        保存scaler对象

        Parameters:
        -----------
        path : str, optional
            保存路径，如果为None则使用默认路径
        """
        if path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(project_root, 'outputs', 'scaler.pkl')

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        print(f"Scaler已保存到: {path}")

    def load_scaler(self, path=None):
        """
        加载scaler对象

        Parameters:
        -----------
        path : str, optional
            加载路径，如果为None则使用默认路径
        """
        if path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(project_root, 'outputs', 'scaler.pkl')

        self.scaler = joblib.load(path)
        print(f"Scaler已从 {path} 加载")

    def get_data_summary(self):
        """
        获取数据摘要

        Returns:
        --------
        dict: 数据摘要信息
        """
        if self.X is None:
            self.load_data()

        summary = {
            'n_samples': self.X.shape[0],
            'n_features': self.X.shape[1],
            'feature_names': self.feature_names,
            'class_distribution': {
                'Rock (R)': int(np.sum(self.y == 0)),
                'Mine (M)': int(np.sum(self.y == 1))
            },
            'data_mean': float(self.X.mean()),
            'data_std': float(self.X.std())
        }

        if self.X_scaled is not None:
            summary['scaled_mean'] = float(self.X_scaled.mean())
            summary['scaled_std'] = float(self.X_scaled.std())

        return summary

def main():
    """主函数：测试数据预处理模块"""
    print("=" * 50)
    print("数据预处理模块测试")
    print("=" * 50)

    # 创建预处理器
    preprocessor = DataPreprocessor()

    # 加载数据
    X, y, feature_names = preprocessor.load_data()

    # 标准化数据
    X_scaled = preprocessor.standardize()

    # 获取数据摘要
    summary = preprocessor.get_data_summary()

    print("\n数据摘要:")
    for key, value in summary.items():
        if key == 'feature_names':
            print(f"  {key}: 共{len(value)}个特征")
        elif key == 'class_distribution':
            print(f"  {key}:")
            for cls, count in value.items():
                print(f"    {cls}: {count}")
        else:
            print(f"  {key}: {value}")

    # 保存scaler
    preprocessor.save_scaler()

    print("\n数据预处理模块测试完成！")

if __name__ == "__main__":
    main()