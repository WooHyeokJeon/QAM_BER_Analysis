import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt #포락선 검출과 LPF 적용을 위해 사용

#맥에서 폰트가 깨져서 한글 폰트 설정 해주기
plt.rcParams['font.family'] = 'AppleGothic'
#y축의 값을 나타낼 때 마이너스(-) 기호 쓰면 오류나서 하이픈으로 대체
plt.rcParams['axes.unicode_minus'] = False

step = 0.001
t = np.arange(0, 1, step)
f = 2
fc = 30
kf = 50

sig = np.sin(2 * np.pi * f * t)
int_sig = np.cumsum(sig) * step

fm_sig = np.cos(2 * np.pi * fc * t + kf * int_sig)

d_fm_sig = np.diff(fm_sig)
d_fm_sig = np.concatenate([[0], d_fm_sig])

env_fm_sig = np.abs(hilbert(d_fm_sig))
env_fm_sig_scaled = env_fm_sig * (np.max(np.abs(sig)) / np.max(np.abs(env_fm_sig)))

#LPF
b, a = butter(5, 0.1)
lpf_fm_sig = filtfilt(b, a, env_fm_sig_scaled)

#IFFT를 이용해 시간 영역에서 복원
ifft_fm_sig = np.fft.ifft(np.fft.fft(lpf_fm_sig))
restored_sig = np.abs(ifft_fm_sig) - np.mean(np.abs(ifft_fm_sig))

plt.figure(figsize=(10, 12))

#1번 그래프
gr1 = plt.subplot(6, 1, 1)
plt.plot(t, sig)

#2번 그래프
gr2 = plt.subplot(6, 1, 2)
plt.plot(t, fm_sig)

#3번 그래프
gr3 = plt.subplot(6, 1, 3)
plt.plot(t, d_fm_sig)

#4번 그래프
gr4 = plt.subplot(6, 1, 4)
plt.plot(t, env_fm_sig_scaled)

#5번 그래프
f_env_fm_sig = np.fft.fftshift(np.fft.fft(env_fm_sig_scaled))
freqs = np.fft.fftshift(np.fft.fftfreq(len(env_fm_sig_scaled), d=step))
gr5 = plt.subplot(6, 1, 5)
plt.plot(freqs, np.abs(f_env_fm_sig))  # 주파수 축에 맞춰 그리기

#6번 그래프
gr6 = plt.subplot(6, 1, 6)
plt.plot(t, restored_sig)
plt.xlabel("시간 (s)")

#그래프들 간격 조정
plt.subplots_adjust(hspace=0.7)

plt.show()