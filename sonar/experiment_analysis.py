# debug_confidence.py
import joblib
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing.scaler import DataPreprocessor

# 加载真实数据
preprocessor = DataPreprocessor()
X, y, _ = preprocessor.load_data()
X_scaled = preprocessor.standardize()

# 加载模型
model = joblib.load('models/best_mlp_model.pkl')
scaler = joblib.load('models/scaler.pkl')

print("="*60)
print("置信度对比测试")
print("="*60)

# 1. 真实样本（标准化后的）
real_sample = X_scaled[0:1]
proba_real = model.predict_proba(real_sample)[0]
print(f"\n【真实样本】")
print(f"  特征范围: [{real_sample.min():.3f}, {real_sample.max():.3f}]")
print(f"  岩石概率: {proba_real[0]:.6f}")
print(f"  矿石概率: {proba_real[1]:.6f}")
print(f"  置信度: {max(proba_real)*100:.2f}%")

# 2. 假数据（原始值0.03-0.42）
fake_raw = np.array([[0.0312, 0.0351, 0.0423, 0.0489, 0.0523, 0.0587, 0.0634, 0.0712, 0.0789, 0.0856,
                       0.0923, 0.0987, 0.1045, 0.1123, 0.1189, 0.1245, 0.1312, 0.1389, 0.1456, 0.1523,
                       0.1587, 0.1645, 0.1712, 0.1789, 0.1856, 0.1923, 0.1987, 0.2045, 0.2112, 0.2189,
                       0.2256, 0.2323, 0.2387, 0.2445, 0.2512, 0.2589, 0.2656, 0.2723, 0.2787, 0.2845,
                       0.2912, 0.2989, 0.3056, 0.3123, 0.3187, 0.3245, 0.3312, 0.3389, 0.3456, 0.3523,
                       0.3587, 0.3645, 0.3712, 0.3789, 0.3856, 0.3923, 0.3987, 0.4045, 0.4112, 0.4189]])
fake_scaled = scaler.transform(fake_raw)
proba_fake = model.predict_proba(fake_scaled)[0]

print(f"\n【假数据（你之前用的）】")
print(f"  特征范围: [{fake_scaled.min():.3f}, {fake_scaled.max():.3f}]")
print(f"  岩石概率: {proba_fake[0]:.6f}")
print(f"  矿石概率: {proba_fake[1]:.6f}")
print(f"  置信度: {max(proba_fake)*100:.2f}%")

# 3. 检查模型内部输出
print(f"\n【诊断】")
print(f"  假数据标准化后的值异常大，导致softmax饱和")
print(f"  这是因为我构造的假数据不符合真实数据分布")