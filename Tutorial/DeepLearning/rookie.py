#%%

"""Softmax."""
import numpy as np

scores = [1.0, 2.0, 3.0]
# scores = np.array([[1, 2, 3, 6],
#                    [2, 4, 5, 6],
#                    [3, 8, 7, 6]])

def softmax(scores):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(scores) / np.sum(np.exp(scores), axis=0)

print(softmax(scores))

#%%
# Plot softmax curves
import matplotlib.pyplot as plt

x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np

def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

x = np.arange(-10, 10, 0.1)
h = Sigmoid(x)

plt.plot(x, h)
plt.axvline(0.0, color='k')  # 坐标轴上加一条竖直的线（0位置）
plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
plt.axhline(y=0.5, ls='dotted', color='k')
plt.yticks([0.0, 0.5, 1.0])  # y轴标度
plt.ylim(-0.1, 1.1)  # y轴范围
plt.show()

