#AWGN
import numpy as np
import matplotlib.pyplot as plt

size=100000
real_noise=np.random.randn(size)
#복소수 신호의 절대값 후 제곱 한 다음에 평균내면 그게 파워임. 그냥 이렇게 정의 함.
noise_power=np.mean(np.abs(real_noise)**2)
print(noise_power)
imag_noise=np.random.randn(size)
awgn=(real_noise+imag_noise*1j)/np.sqrt(2)
awgn_power=np.mean(np.abs(awgn)**2)
print(awgn_power)

plt.figure(figsize=(3,3))
plt.plot(awgn.real,awgn.imag,'.')
plt.show()
