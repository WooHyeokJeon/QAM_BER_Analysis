#AM 그래프 안 그려지는 거 다시 확인하기
import numpy as np
import matplotlib.pyplot as plt

time_step = 0.002
t=np.arange(0,1,time_step)
sig= 3*np.sin(2*np.pi*2*t)+1*np.sin(2*np.pi*4*t)+2*np.sin(2*np.pi*6*t)
plt.subplot(911)
plt.plot(t,sig)

carrier = np.cos(2*np.pi*100*t)
plt.subplot(912)
plt.plot(t,carrier)

am_sig = sig*carrier
plt.subplot(913)
plt.plot(t,am_sig)

f_sig = np.fft.fftshift(np.fft.fft(sig)) #fftshift안 하면 원하는 그림 못 얻음
plt.subplot(914)
plt.plot(np.abs(f_sig)) #복소수가 있어 절대값


f_am_sig = np.fft.fftshift(np.fft.fft(am_sig))
plt.subplot(915)
plt.plot(t,abs(f_am_sig))

dam_am_sig = am_sig * carrier
plt.subplot(916)
plt.plot(t,dam_am_sig)

f_dam_am_sig = np.fft.fftshift(np.fft.fft(dam_am_sig))
plt.subplot(917)
plt.plot(t,np.abs(f_dam_am_sig))

lpf = np.zeros((500))
lpf[200:300]=1  #246부터 254까지 필터를 좁게 만들어보기. 이렇게 되면 6헤르프 짜리 필터가 잘려 나감
                #48:45까지 하면 고주파 성분이 일부 포함됨. 즉 이거 두 개 하면 기존의 신호와는 다른 신호가 얻어지는 것을 볼 수 있음.

lpf_f_sig = f_dam_am_sig * lpf
plt.subplot(918)
plt.plot(np.abs(lpf_f_sig))

dam_am_sig = np.fft.ifft(np.fft.fftshift(lpf_f_sig))
plt.subplot(919)
plt.plot(dam_am_sig)




plt.show()