#!/usr/bin/env python3
"""
随机搜索优化MLP - 预测学生是否辍学
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import uniform, randint
import warnings

warnings.filterwarnings('ignore')


def load_data(file_path='dataset/data.csv'):
    """加载数据"""
    print("加载数据...")

    df = pd.read_csv(file_path, sep=';', encoding='utf-8')
    df.columns = df.columns.str.replace('"', '').str.strip()
    df.columns = df.columns.str.replace('\t', '')

    X = df.drop(columns=['Target'])
    y = (df['Target'] == 'Dropout').astype(int)

    # 预处理
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = X[col].astype(str).str.replace(',', '.').astype(float)
            except:
                pass

    cat_cols = X.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols)

    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"数据: {X_scaled.shape[0]} 样本, {X_scaled.shape[1]} 特征")
    print(f"辍学: {y.mean() * 100:.1f}%, 不辍学: {(1 - y.mean()) * 100:.1f}%")

    return X_scaled, y, scaler


def random_search_optimize(X, y):
    """随机搜索最佳参数"""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 类别权重
    weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    sample_weight = np.array([weights[int(cls)] for cls in y_train])
    print(f"类别权重: 不辍学={weights[0]:.3f}, 辍学={weights[1]:.3f}")

    # 参数搜索空间
    param_dist = {
        'hidden_layer_sizes': [
            (100,), (200,), (300,),
            (100, 50), (200, 100), (300, 150),
            (100, 50, 25), (200, 100, 50), (300, 150, 75),
            (500,), (500, 250),
        ],
        'alpha': uniform(0.00001, 0.001),  # L2正则化
        'learning_rate_init': uniform(0.0001, 0.005),  # 学习率
        'batch_size': [32, 64, 128, 256],
        'activation': ['relu', 'tanh'],
        'max_iter': [500, 800, 1000],
    }

    print("\n开始随机搜索...")

    mlp = MLPClassifier(
        early_stopping=True,
        random_state=42,
        verbose=False
    )

    random_search = RandomizedSearchCV(
        mlp, param_dist,
        n_iter=50,  # 尝试50种组合
        cv=3,
        scoring='f1',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X_train, y_train, sample_weight=sample_weight)

    print(f"\n最佳参数: {random_search.best_params_}")
    print(f"最佳CV分数: {random_search.best_score_:.4f}")

    # 测试集评估
    y_pred = random_search.best_estimator_.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    print(f"\n测试集F1: {f1:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['不辍学', '辍学']))

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n混淆矩阵:")
    print(f"           预测不辍学  预测辍学")
    print(f"实际不辍学:     {cm[0, 0]:4d}       {cm[0, 1]:4d}")
    print(f"实际辍学:       {cm[1, 0]:4d}       {cm[1, 1]:4d}")

    return random_search.best_estimator_, f1, random_search.best_params_


def main():
    X, y, scaler = load_data('dataset/data.csv')

    model, f1, best_params = random_search_optimize(X, y)

    print(f"\n{'=' * 50}")
    print(f"最终F1: {f1:.4f}")

    if f1 >= 0.85:
        print("🎉 达成85%！")
    elif f1 >= 0.82:
        print(f"📈 当前最佳: {f1:.4f}，接近85%")
    else:
        print(f"📈 当前最佳: {f1:.4f}")

    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/best_dropout_mlp.pkl')
    print("✅ 模型已保存")


if __name__ == "__main__":
    main()