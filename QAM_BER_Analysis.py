import numpy as np
import matplotlib.pyplot as plt

#QPSK 심볼
QPSK = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)  #평균 전력을 1로 정규화

#16-QAM 심볼
real_imag_values_16 = np.array([-3, -1, 1, 3])
QAM_16 = (real_imag_values_16[:, None] + 1j * real_imag_values_16).flatten() / np.sqrt(10)  #평균 전력을 1로 정규화

#64-QAM 심볼
real_imag_values_64 = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
QAM_64 = (real_imag_values_64[:, None] + 1j * real_imag_values_64).flatten() / np.sqrt(42)  #평균 전력을 1로 정규화

#시뮬레이션
data_size = 100000

#QPSK의 BER 계산
ber_qpsk = []
snr_qpsk = np.arange(0, 41)
for snr_db in snr_qpsk:
    binary_data = np.random.randint(0, 2, (data_size, 2)) #심볼당 2비트 데이터 생성
    binary_to_decimal = np.array([2, 1])
    decimal_indices = np.dot(binary_data, binary_to_decimal)

    tx_symbols = QPSK[decimal_indices]  #전송 심볼 매핑
    noise_power = 10 ** (-snr_db / 10)  #노이즈 전력 계산
    noise_std = np.sqrt(noise_power / 2)  #노이즈 표준편차 계산
    noise = (np.random.randn(data_size) + 1j * np.random.randn(data_size)) * noise_std
    rx_symbols = tx_symbols + noise  #수신 신호 생성

    detected_symbols = np.argmin(np.abs(rx_symbols[:, None] - QPSK[None, :]), axis=1)  #디모듈레이션
    detected_bits = np.column_stack([detected_symbols // 2, detected_symbols % 2])

    bit_errors = np.sum(binary_data != detected_bits)  #비트 오류 계산
    ber_qpsk.append(bit_errors / (data_size * 2))

#16-QAM의 BER 계산
ber_16qam = []
snr_16qam = np.arange(0, 41)
for snr_db in snr_16qam:
    binary_data = np.random.randint(0, 2, (data_size, 4))  #심볼당 4비트 데이터 생성
    binary_to_decimal = np.array([8, 4, 2, 1])
    decimal_indices = np.dot(binary_data, binary_to_decimal)

    tx_symbols = QAM_16[decimal_indices]  #전송 심볼 매핑
    noise_power = 10 ** (-snr_db / 10)  #노이즈 전력 계산
    noise_std = np.sqrt(noise_power / 2)  #노이즈 표준편차 계산
    noise = (np.random.randn(data_size) + 1j * np.random.randn(data_size)) * noise_std
    rx_symbols = tx_symbols + noise  #수신 신호 생성

    detected_symbols = np.argmin(np.abs(rx_symbols[:, None] - QAM_16[None, :]), axis=1)  #디모듈레이션
    detected_bits = np.array([[int(x) for x in format(idx, '04b')] for idx in detected_symbols])

    bit_errors = np.sum(binary_data != detected_bits)  #비트 오류 계산
    ber_16qam.append(bit_errors / (data_size * 4))

#64-QAM의 BER
ber_64qam = []
snr_64qam = np.arange(0, 41)
for snr_db in snr_64qam:
    binary_data = np.random.randint(0, 2, (data_size, 6))  #심볼당 6비트 데이터 생성
    binary_to_decimal = np.array([32, 16, 8, 4, 2, 1])
    decimal_indices = np.dot(binary_data, binary_to_decimal)

    tx_symbols = QAM_64[decimal_indices]  #전송 심볼 매핑
    noise_power = 10 ** (-snr_db / 10)  #노이즈 전력 계산
    noise_std = np.sqrt(noise_power / 2)  #노이즈 표준편차 계산
    noise = (np.random.randn(data_size) + 1j * np.random.randn(data_size)) * noise_std
    rx_symbols = tx_symbols + noise  #수신 신호 생성

    detected_symbols = np.argmin(np.abs(rx_symbols[:, None] - QAM_64[None, :]), axis=1)  #디모듈레이션

    bit_errors = np.sum(decimal_indices != detected_symbols)  #비트 오류 계산
    current_ber = bit_errors / (data_size * 6)  #BER 계산
    ber_64qam.append(current_ber)

    #높은 SNR 구간에서 로그 출력
    if snr_db >= 38:
        print(f"SNR: {snr_db} dB, BER: {current_ber}")

plt.figure(figsize=(10, 6))

#QPSK
plt.semilogy(
    snr_qpsk,
    ber_qpsk,
    marker='D',
    linestyle='-',
    color='purple',
    label='QPSK BER'
)

#16-QAM
plt.semilogy(
    snr_16qam,
    ber_16qam,
    marker='o',
    linestyle='-',
    color='orange',
    label='16-QAM BER'
)

#64-QAM
plt.semilogy(
    snr_64qam,
    ber_64qam,
    marker='s',
    linestyle='-',
    color='teal',
    label='64-QAM BER'
)

plt.title("BER vs. SNR for QPSK, 16-QAM, and 64-QAM (Updated Design)")
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.grid(True, which="both", linestyle="--")
plt.xlim(0, 35)
plt.legend()
plt.show()
