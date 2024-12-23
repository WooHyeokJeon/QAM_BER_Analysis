import numpy as np
import matplotlib.pyplot as plt

min = -10
max = 10
step = 0.01
x = np.arange(min, max, step)
sig = np.sqrt(2) #3.16
g_pr = 1 / np.sqrt(np.pi * 2) * np.exp(-((x-sig) ** 2) / 2)
plt.figure(figsize=(20,3))
plt.plot(x, g_pr)
plt.show()

#전체 확률
sum_prob = np.sum(g_pr * step)
print(sum_prob)

theory_error_prob = np.sum(g_pr[:1000] * step)
print("이론 BER : ", theory_error_prob)

size = 100000
awgn = np.random.randn(size)
rcv_sig = sig + awgn
num_err = np.sum(rcv_sig < 0+0)
bit_error_rate = num_err / size
print("실험 BER : ", bit_error_rate)