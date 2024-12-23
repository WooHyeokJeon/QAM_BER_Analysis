import numpy as np
import matplotlib.pyplot as plt
size=100000
sig=np.exp(1j*np.pi/4)
print(sig)
sig=np.sqrt(10)*sig
print(sig)#power가 10인 신호
real_noise=np.random.randn(size)
imag_noise=np.random.randn(size)
awgn=(real_noise+imag_noise*1j)/np.sqrt(2)
rcv_sig=sig+awgn
plt.figure(figsize=(3,3))
plt.plot(rcv_sig.real,rcv_sig.imag,'.')
plt.show()
print(np.mean(rcv_sig.real))
print(np.mean(rcv_sig.imag))

sig_power=np.abs(sig)**2
noise_power=np.mean(np.abs(awgn)**2)
print("SNR: ",10*np.log10(sig_power/noise_power),"dB")

'''
angle=np.random.rand(size)*2*np.pi
sig=np.exp(1j*angle)
plt.figure(figsize=(3,3))
plt.plot(sig.real,sig.imag,'.')
plt.show()
'''