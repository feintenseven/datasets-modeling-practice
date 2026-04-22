#!/usr/bin/env python3
"""
下载Sonar数据集
"""

import pandas as pd
import numpy as np
import os
import urllib.request

def download_sonar_data():
    """下载Sonar数据集并保存为CSV"""
    print("正在下载Sonar数据集...")

    # Sonar数据集的URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"

    try:
        # 下载数据
        print("从UCI机器学习仓库下载数据...")
        data_path = os.path.join('data', 'sonar.csv')

        # 直接下载CSV文件
        urllib.request.urlretrieve(url, data_path)

        # 读取数据并添加列名
        df = pd.read_csv(data_path, header=None)

        # 添加列名：60个特征 + 1个目标列
        feature_names = [f'feature_{i}' for i in range(60)]
        df.columns = feature_names + ['target']

        # 重新保存为CSV
        df.to_csv(data_path, index=False)

        print(f"数据集已保存到: {data_path}")
        print(f"数据集形状: {df.shape}")
        print(f"特征数量: {df.shape[1] - 1}")
        print(f"样本数量: {df.shape[0]}")
        print(f"类别分布:\n{df['target'].value_counts()}")

        return data_path

    except Exception as e:
        print(f"下载数据集时出错: {e}")

        # 如果下载失败，创建一个示例数据集
        print("创建示例数据集...")
        return create_sample_data()

def create_sample_data():
    """创建示例Sonar数据集"""
    data_path = os.path.join('data', 'sonar.csv')

    # 创建示例数据（60个特征 + 1个标签列）
    n_samples = 208
    n_features = 60

    # 生成随机数据
    np.random.seed()
    X = np.random.randn(n_samples, n_features)

    # 创建简单的非线性关系（模拟Sonar数据的模式）
    # Rock (R) 和 Mine (M) 有不同的频率模式
    for i in range(n_samples):
        if i < 104:  # 前104个样本为Rock
            # Rock模式：低频特征较强
            X[i, :20] += np.random.randn(20) * 0.5 + 0.3
            X[i, 20:40] += np.random.randn(20) * 0.3
            X[i, 40:] += np.random.randn(20) * 0.1
        else:  # 后104个样本为Mine
            # Mine模式：高频特征较强
            X[i, :20] += np.random.randn(20) * 0.1
            X[i, 20:40] += np.random.randn(20) * 0.3
            X[i, 40:] += np.random.randn(20) * 0.5 + 0.3

    # 创建标签
    y = np.array(['R'] * 104 + ['M'] * 104)

    # 转换为DataFrame
    columns = [f'feature_{i}' for i in range(n_features)] + ['target']
    df = pd.DataFrame(np.column_stack([X, y]), columns=columns)

    # 打乱数据
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_csv(data_path, index=False)
    print(f"示例数据集已保存到: {data_path}")
    print(f"数据集形状: {df.shape}")
    print(f"特征数量: {df.shape[1] - 1}")
    print(f"样本数量: {df.shape[0]}")
    print(f"类别分布:\n{df['target'].value_counts()}")

    return data_path

if __name__ == "__main__":
    download_sonar_data()