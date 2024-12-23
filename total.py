import numpy as np
import matplotlib.pyplot as plt

# QPSK Symbols
QPSK = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)  # Normalize power

# 16-QAM Symbols
QAM_16 = np.array([
    1 + 1j, 1 + 3j, 3 + 1j, 3 + 3j,
    1 - 1j, 1 - 3j, 3 - 1j, 3 - 3j,
    -1 + 1j, -1 + 3j, -3 + 1j, -3 + 3j,
    -1 - 1j, -1 - 3j, -3 - 1j, -3 - 3j
]) / np.sqrt(10)  # Normalize power

# 64-QAM Symbols
QAM_64 = np.array([
    -7 - 7j, -7 - 5j, -7 - 3j, -7 - 1j,
    -7 + 1j, -7 + 3j, -7 + 5j, -7 + 7j,
    -5 + 7j, -5 + 5j, -5 + 3j, -5 + 1j,
    -5 - 1j, -5 - 3j, -5 - 5j, -5 - 7j,
    -3 - 7j, -3 - 5j, -3 - 3j, -3 - 1j,
    -3 + 1j, -3 + 3j, -3 + 5j, -3 + 7j,
    -1 + 7j, -1 + 5j, -1 + 3j, -1 + 1j,
    -1 - 1j, -1 - 3j, -1 - 5j, -1 - 7j,
    1 - 7j, 1 - 5j, 1 - 3j, 1 - 1j,
    1 + 1j, 1 + 3j, 1 + 5j, 1 + 7j,
    3 + 7j, 3 + 5j, 3 + 3j, 3 + 1j,
    3 - 1j, 3 - 3j, 3 - 5j, 3 - 7j,
    5 - 7j, 5 - 5j, 5 - 3j, 5 - 1j,
    5 + 1j, 5 + 3j, 5 + 5j, 5 + 7j,
    7 + 7j, 7 + 5j, 7 + 3j, 7 + 1j,
    7 - 1j, 7 - 3j, 7 - 5j, 7 - 7j
]) / np.sqrt(42)  # Normalize power

# Simulation Parameters
data_size = 100000  # Number of symbols

# BER for QPSK
ber_qpsk = []
snr_qpsk = np.arange(0, 41)
for snr_db in snr_qpsk:
    binary_data = np.random.randint(0, 2, (data_size, 2))  # 2 bits per symbol
    binary_to_decimal = np.array([2, 1])
    decimal_indices = np.dot(binary_data, binary_to_decimal)

    tx_symbols = QPSK[decimal_indices]
    noise_power = 10 ** (-snr_db / 10)
    noise_std = np.sqrt(noise_power / 2)
    noise = (np.random.randn(data_size) + 1j * np.random.randn(data_size)) * noise_std
    rx_symbols = tx_symbols + noise

    detected_symbols = np.argmin(np.abs(rx_symbols[:, None] - QPSK[None, :]), axis=1)
    detected_bits = np.column_stack([detected_symbols // 2, detected_symbols % 2])

    bit_errors = np.sum(binary_data != detected_bits)
    ber_qpsk.append(bit_errors / (data_size * 2))

# BER for 16-QAM
ber_16qam = []
snr_16qam = np.arange(0, 41)
for snr_db in snr_16qam:
    binary_data = np.random.randint(0, 2, (data_size, 4))  # 4 bits per symbol
    binary_to_decimal = np.array([8, 4, 2, 1])
    decimal_indices = np.dot(binary_data, binary_to_decimal)

    tx_symbols = QAM_16[decimal_indices]
    noise_power = 10 ** (-snr_db / 10)
    noise_std = np.sqrt(noise_power / 2)
    noise = (np.random.randn(data_size) + 1j * np.random.randn(data_size)) * noise_std
    rx_symbols = tx_symbols + noise

    detected_symbols = np.argmin(np.abs(rx_symbols[:, None] - QAM_16[None, :]), axis=1)
    detected_bits = np.array([[int(x) for x in format(idx, '04b')] for idx in detected_symbols])

    bit_errors = np.sum(binary_data != detected_bits)
    ber_16qam.append(bit_errors / (data_size * 4))

# BER for 64-QAM
ber_64qam = []
snr_64qam = np.arange(0, 41)
for snr_db in snr_64qam:
    binary_data = np.random.randint(0, 2, (data_size, 6))  # 6 bits per symbol
    binary_to_decimal = np.array([32, 16, 8, 4, 2, 1])
    decimal_indices = np.dot(binary_data, binary_to_decimal)

    tx_symbols = QAM_64[decimal_indices]
    noise_power = 10 ** (-snr_db / 10)
    noise_std = np.sqrt(noise_power / 2)
    noise = (np.random.randn(data_size) + 1j * np.random.randn(data_size)) * noise_std
    rx_symbols = tx_symbols + noise

    detected_symbols = np.argmin(np.abs(rx_symbols[:, None] - QAM_64[None, :]), axis=1)

    bit_errors = np.sum(decimal_indices != detected_symbols)
    ber_64qam.append(bit_errors / (data_size * 6))

# Plot the BER for QPSK, 16-QAM, and 64-QAM
plt.figure(figsize=(10, 6))

# QPSK
plt.semilogy(
    snr_qpsk,
    ber_qpsk,
    marker='D',
    linestyle='-',
    color='purple',
    label='QPSK BER'
)

# 16-QAM
plt.semilogy(
    snr_16qam,
    ber_16qam,
    marker='o',
    linestyle='-',
    color='orange',
    label='16-QAM BER'
)

# 64-QAM
plt.semilogy(
    snr_64qam,
    ber_64qam,
    marker='s',
    linestyle='-',
    color='teal',
    label='64-QAM BER'
)

# Add labels, grid, and legend
plt.title("BER vs. SNR for QPSK, 16-QAM, and 64-QAM (Updated Design)")
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.grid(True, which="both", linestyle="--")
plt.xlim(0, 35)  # Limit x-axis range to 35
plt.legend()
plt.show()
