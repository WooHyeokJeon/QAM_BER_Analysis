import numpy as np
import matplotlib.pyplot as plt

QAM_64 = [
    -7 - 7j, -7 - 5j, -7 - 3j, -7 - 1j,
    -7 + 1j, -7 + 3j, -7 + 5j, -7 + 7j,

    -5 + 7j, -5 + 5j, -5 + 3j, -5 + 1j,
    -5 - 1j, -5 - 3j, -5 - 5j, -5 - 7j,

    -3 - 7j, -3 - 5j, -3 - 3j, -3 - 1j,
    -3 + 1j, -3 + 3j, -3 + 5j, -3 + 7j,

    -1 + 7j, -1 + 5j, -1 + 3j, -1 + 1j,
    -1 - 1j, -1 - 3j, -1 - 5j, -1 - 7j,

    1 - 7j, 1 - 5j, 1 - 3j, 1 - 1j,
    1 + 1j,  1 + 3j,  1 + 5j,  1 + 7j,

    3 + 7j, 3 + 5j, 3 + 3j, 3 + 1j,
    3 - 1j, 3 - 3j, 3 - 5j, 3 - 7j,

    5 - 7j,  5 - 5j,  5 - 3j,  5 - 1j,
    5 + 1j,  5 + 3j,  5 + 5j,  5 + 7j,

    7 + 7j, 7 + 5j, 7 + 3j, 7 + 1j,
    7 - 1j, 7 - 3j, 7 - 5j, 7 - 7j,
    ] / np.sqrt(42)  # 평균 전력을 정규화

QAM_to_bit = [
    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 0, 0],

    [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 0],
    [0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1], [0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 0],

    [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 0, 1], [0, 1, 1, 1, 0, 0],

    [0, 1, 0, 1, 0, 0], [0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 1, 1], [0, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 1], [0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0],

    [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 0],
    [1, 1, 0, 1, 1, 0], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 0],

    [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0],
    [1, 1, 1, 0, 1, 0], [1, 1, 1, 0, 1, 1], [1, 1, 1, 0, 0, 1], [1, 1, 1, 0, 0, 0],

    [1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 1], [1, 0, 1, 0, 1, 0],
    [1, 0, 1, 1, 1, 0], [1, 0, 1, 1, 1, 1], [1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 0, 0],

    [1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 1, 1], [1, 0, 0, 1, 1, 0],
    [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 1, 1], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0],
]


data_size = 100000
snr_db_range = range(0, 26, 2)  # SNR 범위 (0~25 dB)
ser = []  # SER 저장
ber = []  # BER 저장

for snr_db in snr_db_range:
    binary_data = np.random.randint(0, 2, (data_size, 6))
    binary_to_decimal = np.array([32, 16, 8, 4, 2, 1])
    decimal_indices = np.dot(binary_data, binary_to_decimal)

    tx_symbols = QAM_64[decimal_indices]

    noise_std = 10 ** (-snr_db / 20)
    noise = (np.random.randn(data_size) + 1j * np.random.randn(data_size)) * noise_std / np.sqrt(2)
    rx_symbols = tx_symbols + noise

    detected_symbols = np.argmin(np.abs(rx_symbols[:, None] - QAM_64[None, :]), axis=1)

    symbol_errors = np.sum(decimal_indices != detected_symbols)
    current_ser = symbol_errors / data_size
    ser.append(current_ser)

    current_ber = current_ser / 6
    ber.append(current_ber)

plt.figure(figsize=(10, 6))
plt.semilogy(snr_db_range, ber, marker='o', color='red', label='64-QAM BER')
plt.title("64-QAM BER vs SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.show()