# Sonar 数据集：LASSO + MLP 分类项目

## 一、项目目标

构建完整机器学习流程：

数据预处理 → 特征选择（LASSO）→ MLP建模 → 模型评估 → 可视化展示

核心问题：

* 在小样本高维数据中，LASSO 是否提升模型表现？
* MLP 是否优于传统线性模型？

---

## 二、数据集说明

* 数据集：Sonar
* 样本数：208
* 特征数：60（连续数值）
* 任务：二分类（R = Rock, M = Mine）

特点：

* 小样本
* 高维
* 非线性结构明显

---

## 三、项目结构

```
project/
│
├── data/
│   └── sonar.csv
│
├── preprocessing/
│   ├── scaler.py
│   ├── feature_selection.py
│
├── models/
│   └── mlp_model.py
│
├── pipeline/
│   └── main_pipeline.py
│
├── evaluation/
│   └── metrics.py
│
├── app/
│   └── streamlit_app.py
│
└── outputs/
    ├── model.pkl
    ├── selected_features.json
```

---

## 四、模块设计

---

### 1. 数据预处理模块

功能：

* 读取数据
* 标签编码
* 标准化

示例代码：

```
X = df.iloc[:, :-1]
y = df.iloc[:, -1].map({'R': 0, 'M': 1})

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

关键点：

* MLP 对输入尺度敏感 → 必须标准化

---

### 2. 特征选择模块（LASSO）

方法：

* Logistic Regression + L1 正则

示例代码：

```
from sklearn.linear_model import LogisticRegression
import numpy as np

lasso = LogisticRegression(penalty='l1', solver='liblinear')
lasso.fit(X_scaled, y)

selected_idx = np.where(lasso.coef_[0] != 0)[0]
X_selected = X_scaled[:, selected_idx]
```

输出：

* 特征索引
* 降维后的数据

必须记录：

* 原始维度：60
* 筛选后维度：N（如 15~25）

---

### 3. MLP 模型模块（核心）

模型定义：

```
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=16,
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    random_state=42
)
```

训练流程：

```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
```

关键设计：

* 小样本 → 浅层网络
* ReLU 激活
* L2 正则（alpha）
* early stopping 防过拟合

---

### 4. 评估模块

示例代码：

```
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
```

必须指标：

* Accuracy
* Precision
* Recall
* F1-score

加分项：

```
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_selected, y, cv=5)
```

---

### 5. 实验设计（重点）

实验1：LASSO效果

| 方法          | Accuracy |
| ----------- | -------- |
| MLP（原始60维）  | ?        |
| MLP（LASSO后） | ?        |

实验2：结构对比

| 结构      | Accuracy |
| ------- | -------- |
| (16,)   | ?        |
| (32,16) | ?        |
| (64,32) | ?        |

实验3（可选）：正则化影响

| alpha  | Accuracy |
| ------ | -------- |
| 0.0001 | ?        |
| 0.001  | ?        |
| 0.01   | ?        |

---

### 6. 可视化模块（Streamlit）

功能：

* 用户输入特征
* 模型预测
* 展示结果

示例代码：

```
import streamlit as st

st.title("Sonar Classification")

input_data = st.text_area("输入60维特征（逗号分隔）")

if st.button("预测"):
    data = process_input(input_data)
    pred = model.predict(data)
    st.write("预测结果:", pred)
```

展示内容：

* Accuracy
* 特征数量
* 输入 → 输出

---

## 五、核心亮点

1. 方法组合

   * LASSO（降维） + MLP（非线性建模）

2. 小样本优化

   * 正则化
   * early stopping
   * 控制模型复杂度

3. 实验设计完整

   * 特征选择对比
   * 网络结构对比

4. 工程化实现

   * 模块化 pipeline
   * GUI 展示

---

## 六、报告结构建议

1. 引言
2. 数据与预处理
3. 特征选择方法（LASSO）
4. 模型设计（MLP）
5. 实验与结果分析
6. 系统实现（GUI）
7. 总结

---

## 七、总结

本项目在小样本高维数据场景下：

* 使用 LASSO 进行特征筛选
* 构建轻量级 MLP 模型
* 结合正则化与早停机制

实现了较好的分类性能，并验证了特征选择与非线性建模的有效性。
