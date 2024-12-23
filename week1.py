import numpy as np
import matplotlib.pyplot as plt

signal = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
f_sig = np.fft.fft(signal) # 복소수가 나와 이대로 그릴 수 없음
f_sig = np.fft.fftshift(f_sig) # shift를 안 하면 플러스 주파수 그리고 오른쪽에 마이너스를 그려서 옮기는 거
plt.plot(np.abs(f_sig))
plt.show()