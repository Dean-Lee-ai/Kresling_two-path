import numpy as np
import pandas as pd
import plotly.express as px

# 生成数据
x = np.linspace(-1, 1, 20)
y = np.linspace(-1, 1, 20)
z = np.linspace(-1, 1, 20)
X, Y, Z = np.meshgrid(x, y, z)
w_fixed = 0.5  # 固定 w
F = np.sin(X) + np.cos(Y) + np.tanh(Z) + w_fixed  # 示例函数

# 转换为 DataFrame
df = pd.DataFrame({"x": X.flatten(), "y": Y.flatten(), "z": Z.flatten(), "f(x,y,z,w)": F.flatten()})

# 3D 交互式可视化
fig = px.scatter_3d(df, x="x", y="y", z="z", color="f(x,y,z,w)", opacity=0.8)
fig.show()
