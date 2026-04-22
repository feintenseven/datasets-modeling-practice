#!/usr/bin/env python3
"""
Streamlit应用
功能：交互式GUI用于模型预测和结果展示
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置页面配置
st.set_page_config(
    page_title="Sonar Classification",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用标题
st.title("🔊 Sonar数据集分类系统")
st.markdown("---")

# 初始化session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'input_history' not in st.session_state:
    st.session_state.input_history = []

class SonarApp:
    """Sonar分类应用"""

    def __init__(self):
        """初始化应用"""
        self.model = None
        self.scaler = None
        self.selector = None
        self.feature_names = None
        self.selected_features = None
        self.selected_indices = None
        self.load_models()

    def load_models(self):
        """加载模型和预处理对象"""
        try:
            # 加载scaler
            scaler_path = os.path.join('outputs', 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                st.sidebar.success("✓ Scaler加载成功")
            else:
                st.sidebar.warning("Scaler文件未找到")

            # 加载特征选择信息
            features_path = os.path.join('outputs', 'selected_features.json')
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    features_data = json.load(f)
                self.selected_indices = features_data['selected_indices']
                self.selected_features = features_data['selected_features']
                st.sidebar.success("✓ 特征选择信息加载成功")
            else:
                st.sidebar.warning("特征选择文件未找到")

            # 加载最新的MLP模型
            model_files = [f for f in os.listdir('outputs') if f.startswith('mlp_model_') and f.endswith('.pkl')]
            if model_files:
                latest_model = sorted(model_files)[-1]  # 获取最新的模型
                model_path = os.path.join('outputs', latest_model)
                self.model = joblib.load(model_path)
                st.session_state.model_loaded = True
                st.sidebar.success(f"✓ MLP模型加载成功: {latest_model}")
            else:
                st.sidebar.error("未找到MLP模型文件")

            # 加载特征名称
            data_path = os.path.join('data', 'sonar.csv')
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                self.feature_names = df.columns[:-1].tolist()
                st.sidebar.success("✓ 特征名称加载成功")

        except Exception as e:
            st.sidebar.error(f"加载模型时出错: {str(e)}")

    def predict(self, input_features):
        """
        使用加载的模型进行预测

        Parameters:
        -----------
        input_features : array-like
            输入特征

        Returns:
        --------
        dict: 预测结果
        """
        if self.model is None or self.scaler is None or self.selected_indices is None:
            return {"error": "模型未加载"}

        try:
            # 转换为numpy数组
            features = np.array(input_features).reshape(1, -1)

            # 标准化
            features_scaled = self.scaler.transform(features)

            # 特征选择
            features_selected = features_scaled[:, self.selected_indices]

            # 预测
            prediction = self.model.predict(features_selected)[0]
            probability = self.model.predict_proba(features_selected)[0]

            # 获取类别标签
            class_label = "Rock (R)" if prediction == 0 else "Mine (M)"
            class_prob = probability[prediction]

            return {
                "prediction": prediction,
                "class_label": class_label,
                "probability": class_prob,
                "probabilities": probability.tolist(),
                "features_used": len(self.selected_indices)
            }

        except Exception as e:
            return {"error": f"预测时出错: {str(e)}"}

    def display_prediction_result(self, result):
        """显示预测结果"""
        if "error" in result:
            st.error(result["error"])
            return

        # 创建结果容器
        result_container = st.container()
        with result_container:
            st.markdown("### 📊 预测结果")

            # 使用列布局
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("预测类别", result["class_label"])

            with col2:
                st.metric("置信度", f"{result['probability']:.2%}")

            with col3:
                st.metric("使用特征数", result["features_used"])

            # 显示概率分布
            st.markdown("#### 类别概率分布")
            prob_df = pd.DataFrame({
                "类别": ["Rock (R)", "Mine (M)"],
                "概率": result["probabilities"]
            })

            # 使用条形图显示概率
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(prob_df["类别"], prob_df["概率"], color=['skyblue', 'lightcoral'])
            ax.set_ylabel("概率")
            ax.set_ylim([0, 1])
            ax.set_title("类别概率分布")

            # 在柱子上添加数值
            for bar, prob in zip(bars, prob_df["概率"]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                       f'{prob:.2%}', ha='center', va='bottom')

            st.pyplot(fig)

            # 添加解释
            if result["prediction"] == 0:
                st.info("🔍 **解释**: 模型预测为岩石(Rock)。这意味着声纳信号特征更符合岩石的反射模式。")
            else:
                st.info("🔍 **解释**: 模型预测为水雷(Mine)。这意味着声纳信号特征更符合金属物体的反射模式。")

        return result_container

def main():
    """主函数：运行Streamlit应用"""
    # 创建应用实例
    app = SonarApp()

    # 侧边栏
    st.sidebar.title("⚙️ 控制面板")
    st.sidebar.markdown("---")

    # 模型状态
    st.sidebar.subheader("模型状态")
    if st.session_state.model_loaded:
        st.sidebar.success("✅ 模型已加载")
    else:
        st.sidebar.error("❌ 模型未加载")

    # 数据统计
    st.sidebar.subheader("数据统计")
    if app.feature_names:
        st.sidebar.write(f"原始特征数: {len(app.feature_names)}")
    if app.selected_features:
        st.sidebar.write(f"选择特征数: {len(app.selected_features)}")
        reduction = (1 - len(app.selected_features)/len(app.feature_names))*100 if app.feature_names else 0
        st.sidebar.write(f"特征减少: {reduction:.1f}%")

    # 主要区域
    st.header("🎯 实时预测")

    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📝 手动输入", "📁 文件上传", "📊 数据分析", "📈 模型信息"])

    with tab1:
        st.subheader("手动输入特征值")

        # 创建输入表单
        with st.form("prediction_form"):
            st.write("请输入60个特征值（0-1之间的浮点数）：")

            # 如果特征名称已加载，显示特征名
            if app.feature_names:
                # 创建3列，每列20个特征输入
                cols = st.columns(3)
                input_features = []

                for i, feature_name in enumerate(app.feature_names):
                    col_idx = i % 3
                    with cols[col_idx]:
                        # 设置默认值（基于特征索引的简单模式）
                        default_value = 0.1 + (i % 10) * 0.08
                        value = st.number_input(
                            f"{feature_name}",
                            min_value=0.0,
                            max_value=1.0,
                            value=min(default_value, 1.0),
                            step=0.01,
                            key=f"feature_{i}"
                        )
                        input_features.append(value)
            else:
                # 如果没有特征名称，使用简单输入
                input_features = []
                cols = st.columns(3)
                for i in range(60):
                    col_idx = i % 3
                    with cols[col_idx]:
                        default_value = 0.1 + (i % 10) * 0.08
                        value = st.number_input(
                            f"特征 {i}",
                            min_value=0.0,
                            max_value=1.0,
                            value=min(default_value, 1.0),
                            step=0.01,
                            key=f"feature_{i}"
                        )
                        input_features.append(value)

            # 表单提交按钮
            submitted = st.form_submit_button("🚀 开始预测")

            if submitted:
                if len(input_features) != 60:
                    st.error("请输入60个特征值")
                else:
                    with st.spinner("正在预测..."):
                        result = app.predict(input_features)
                        app.display_prediction_result(result)

                        # 保存到历史记录
                        st.session_state.predictions.append(result)
                        st.session_state.input_history.append(input_features)

    with tab2:
        st.subheader("文件上传预测")

        uploaded_file = st.file_uploader("上传CSV文件", type=['csv'])

        if uploaded_file is not None:
            try:
                # 读取CSV文件
                df = pd.read_csv(uploaded_file)

                # 检查数据格式
                if len(df.columns) != 60:
                    st.error(f"文件应包含60个特征列，当前有{len(df.columns)}列")
                else:
                    st.success(f"成功读取 {len(df)} 行数据")

                    # 显示数据预览
                    st.write("数据预览:")
                    st.dataframe(df.head())

                    # 批量预测
                    if st.button("🔍 批量预测"):
                        predictions = []
                        progress_bar = st.progress(0)

                        for idx, row in df.iterrows():
                            features = row.values.tolist()
                            result = app.predict(features)
                            predictions.append(result)

                            # 更新进度条
                            progress_bar.progress((idx + 1) / len(df))

                        # 显示预测结果
                        results_df = pd.DataFrame([
                            {
                                "样本": i+1,
                                "预测类别": "Rock (R)" if p["prediction"] == 0 else "Mine (M)",
                                "置信度": f"{p['probability']:.2%}",
                                "Rock概率": f"{p['probabilities'][0]:.2%}",
                                "Mine概率": f"{p['probabilities'][1]:.2%}"
                            }
                            for i, p in enumerate(predictions) if "error" not in p
                        ])

                        st.write("预测结果:")
                        st.dataframe(results_df)

                        # 统计信息
                        rock_count = sum(1 for p in predictions if "error" not in p and p["prediction"] == 0)
                        mine_count = sum(1 for p in predictions if "error" not in p and p["prediction"] == 1)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Rock (R) 数量", rock_count)
                        with col2:
                            st.metric("Mine (M) 数量", mine_count)

                        # 下载结果
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 下载预测结果",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"处理文件时出错: {str(e)}")

    with tab3:
        st.subheader("数据分析")

        if not st.session_state.model_loaded:
            st.warning("请先加载模型以查看数据分析")
        else:
            # 加载原始数据
            data_path = os.path.join('data', 'sonar.csv')
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)

                # 数据概览
                st.write("### 数据概览")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总样本数", len(df))
                with col2:
                    st.metric("特征数", len(df.columns) - 1)
                with col3:
                    rock_count = sum(df['target'] == 'R')
                    st.metric("Rock (R)", rock_count)
                with col4:
                    mine_count = sum(df['target'] == 'M')
                    st.metric("Mine (M)", mine_count)

                # 类别分布图
                st.write("### 类别分布")
                fig, ax = plt.subplots(figsize=(8, 4))
                df['target'].value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral'])
                ax.set_xlabel("类别")
                ax.set_ylabel("数量")
                ax.set_title("类别分布")
                st.pyplot(fig)

                # 特征统计
                st.write("### 特征统计")
                if st.checkbox("显示特征描述统计"):
                    st.dataframe(df.describe())

                # 相关性分析
                st.write("### 特征相关性")
                if st.checkbox("显示特征相关性热图"):
                    # 计算相关性矩阵
                    corr_matrix = df.iloc[:, :-1].corr()

                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(corr_matrix, ax=ax, cmap='coolwarm', center=0)
                    ax.set_title("特征相关性热图")
                    st.pyplot(fig)

            # 模型性能分析
            st.write("### 模型性能")
            # 加载评估结果
            eval_files = [f for f in os.listdir('outputs') if f.startswith('evaluation_') and f.endswith('.json')]
            if eval_files:
                latest_eval = sorted(eval_files)[-1]
                eval_path = os.path.join('outputs', latest_eval)

                with open(eval_path, 'r') as f:
                    eval_data = json.load(f)

                metrics = eval_data.get('metrics', {})

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("准确率", f"{metrics.get('accuracy', 0):.2%}")
                with col2:
                    st.metric("精确率", f"{metrics.get('precision', 0):.2%}")
                with col3:
                    st.metric("召回率", f"{metrics.get('recall', 0):.2%}")
                with col4:
                    st.metric("F1分数", f"{metrics.get('f1', 0):.2%}")

                if 'roc_auc' in metrics:
                    st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")

    with tab4:
        st.subheader("模型信息")

        if not st.session_state.model_loaded:
            st.warning("模型未加载")
        else:
            # 模型架构信息
            st.write("### 🏗️ 模型架构")
            if hasattr(app.model, 'hidden_layer_sizes'):
                st.write(f"隐藏层结构: {app.model.hidden_layer_sizes}")
            if hasattr(app.model, 'n_layers_'):
                st.write(f"总层数: {app.model.n_layers_}")
            if hasattr(app.model, 'n_outputs_'):
                st.write(f"输出数: {app.model.n_outputs_}")

            # 训练信息
            st.write("### 📚 训练信息")
            if hasattr(app.model, 'n_iter_'):
                st.write(f"训练迭代次数: {app.model.n_iter_}")
            if hasattr(app.model, 'loss_'):
                st.write(f"最终损失: {app.model.loss_:.4f}")

            # 特征选择信息
            st.write("### 🔍 特征选择")
            if app.selected_features:
                st.write(f"选择特征数: {len(app.selected_features)}")
                st.write("选择的特征:")
                features_df = pd.DataFrame({
                    "索引": app.selected_indices,
                    "特征名": app.selected_features
                })
                st.dataframe(features_df)

            # 模型参数
            st.write("### ⚙️ 模型参数")
            if hasattr(app.model, 'get_params'):
                params = app.model.get_params()
                params_df = pd.DataFrame(list(params.items()), columns=["参数", "值"])
                st.dataframe(params_df)

    # 历史记录
    if st.session_state.predictions:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📜 预测历史")
        for i, (pred, inputs) in enumerate(zip(st.session_state.predictions[-5:],
                                              st.session_state.input_history[-5:])):
            if "error" not in pred:
                st.sidebar.write(f"{i+1}. {pred['class_label']} ({pred['probability']:.1%})")

    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Sonar数据集分类系统 | 使用LASSO特征选择和MLP模型</p>
        <p>🔊 识别岩石(Rock)与水雷(Mine)的声纳信号</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()