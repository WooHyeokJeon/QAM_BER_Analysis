import numpy as np
import matplotlib.pyplot as plt

#LC를 이용해서 다이오드 가지고 실제 신호 복원이 가능한지 해보기
time_step = 0.002
t = np.arange(0, 1, time_step)
sig= 3*np.sin(2*np.pi*2*t)+1*np.sin(2*np.pi*4*t)+2*np.sin(2*np.pi*6*t)
sig = sig/6 #이러면 과변조가 아니게 됨
#위 코드 만약 3으로 나눠주게 되면 과변조가 일어남(신호의 최대값은 5인데 3으로 나눠주면 1이 넘게 되니까 -> 그럼 마지막에 신호가 제대로 복원 안 됨)
#과변조가 되면 절대값 취하는 과정에서 6번째 그래프가 원래 밑으로 내려가야 하는데 위로 솟아 오르기 때문에 마지막에 신호 복원이 제대로 안 됨.
#과변조 한 것을 방지하려면 lc앞에 상수를 키워주면 됨. 그럼 원래대로 복원 잘 됨.
plt.subplot(911)
plt.plot(t,sig)

#lc를 사용하면 신호의 Envlop이 잘 보존 된다.
lc = 1*np.cos(2*np.pi*100*t) #맨 앞 계수를 조정하면 0을 기준으로 위 아래로 더 퍼짐.
carrier = 1*np.cos(2*np.pi*100*t)
plt.subplot(912)
plt.plot(t,carrier)

am_sig=sig*carrier+lc
plt.subplot(913)
plt.plot(t,am_sig)

f_sig=np.fft.fftshift(np.fft.fft(sig))
plt.subplot(914)
plt.plot(np.abs(f_sig))

f_am_sig=np.fft.fftshift(np.fft.fft(am_sig))
plt.subplot(915)
plt.plot(t, abs(f_am_sig))

#복조, 다이오드랑 캐페시터를 이용해 절대값 씌우면 됨.
dem_am_sig=np.abs(am_sig)
plt.subplot(916)
plt.plot(t, dem_am_sig)
#스펙트럼 보기
#결과를 보면 절대값만 취했는데 cos을 곱한거랑 같은 효과가 나타남
#(실제로 lc가 있는 경우에 envlop이 그대로 보존 된 상태에서 절대값만 취해주면 cos을 곱해준거랑 완전이 똑같은 일이 일어남)
f_dem_am_sig=np.fft.fftshift(np.fft.fft(dem_am_sig))
plt.subplot(917)
plt.plot(np.abs(f_dem_am_sig))

lpf=np.zeros((500))
lpf[200:300]=1

lpf_f_sig=f_dem_am_sig*lpf
plt.subplot(918)
plt.plot(np.abs(lpf_f_sig))

#원래 신호 복원
#하지만 위로 떠있음. 우리가 carrier을 더해준 만큼 상승. 더해준 값만큼 빼주면 원래 신호 복원 가능
dem_am_sig=np.fft.ifft(np.fft.fftshift(lpf_f_sig))
plt.subplot(919)
plt.plot(dem_am_sig)

plt.show()
