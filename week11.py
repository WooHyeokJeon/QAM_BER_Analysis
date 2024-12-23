#psk
import numpy as np
import matplotlib.pyplot as plt

data_size = 1000000
max_snr = 11 #snr은 0부터 10까지
ber = [] #snr별로 bit error rate(비트 오류율 저장)

for snr_db in range(0, max_snr):
    sig = np.random.randint(0, 2, data_size) * 2 - 1
    noise_pwr =  10**(-snr_db/20) #noise의 평균 크기
    noise = np.random.randn(data_size) * noise_pwr/np.sqrt(2)
    rcv_sig = sig + noise
    detected_sig = ((rcv_sig > 0) + 0) * 2 - 1
    num_error = np.sum(np.abs(sig - detected_sig)/ 2)
    ber.append(num_error / data_size)
snr = np.arange(0, max_snr)
plt.semilogy(snr, ber)
plt.show()

