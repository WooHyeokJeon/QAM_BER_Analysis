#한 분면에 점 찍어서 파워 구하고 -> 점이 어디 찍혔는지 매핑 먼저

import numpy as np
import matplotlib.pyplot as plt

data_size = 16
QAM_16 = [1 + 1j, 1 + 3j, 3 + 1j, 3 + 3j, #1사분면
          1 - 1j, 1 - 3j, 3 - 1j, 3 - 3j, #4사분면
          -1 + 1j, -1 + 3j, -3 + 1j, -3 + 3j, #2사분면
          -1 - 1j, -1 - 3j, -3 - 1j, -3 - 3j #3사분면
          ] / np.sqrt(10)

QAM_to_bit = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],

    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],

    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],

    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
]

print(np.sum(np.abs(QAM_16) ** 2) / 16)

snr_db = 20
binary_data = np.random.randint(0, 2, data_size)
binary_data = binary_data.reshape(-1, 4)
binary_to_10num = np.array([8, 4, 2, 1]).reshape(4, 1)
tmp = np.dot(binary_data, binary_to_10num).transpose() #4비트로 생성한 걸 10진수로 바꿔주는 코드

sym_size = np.shape(tmp)[1]
sym = []
for a in range(sym_size):
    idx = tmp[0, a]
    sym.append(QAM_16[idx])

noise_std = 10**(-snr_db / 20)
real_noise = np.random.randn(sym_size) * noise_std
imag_noise = np.random.randn(sym_size) * noise_std
noise = (real_noise + 1j * imag_noise) / np.sqrt(2)

rcv_sig = sym + noise #노이즈를 심볼 개수만큼 생성하기.

#수신 된 값을 비트로 변환
rcv_binary = []
for a in range(sym_size):
    tmp_min = np.argmin(np.abs(QAM_16 - rcv_sig[a]))
    rcv_binary.append(QAM_to_bit[tmp_min][:])

#에러 센다음 ber 그리기.




import numpy as np
import matplotlib.pyplot as plt

# Define 16-QAM symbols and bit mapping
QAM_16 = np.array([
    1 + 1j, 1 + 3j, 3 + 1j, 3 + 3j,
    1 - 1j, 1 - 3j, 3 - 1j, 3 - 3j,
    -1 + 1j, -1 + 3j, -3 + 1j, -3 + 3j,
    -1 - 1j, -1 - 3j, -3 - 1j, -3 - 3j
]) / np.sqrt(10)  # Normalize power

QAM_to_bit = [
    [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
    [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
    [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
    [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1],
]

# Simulation parameters
data_size = 100000  # Number of bits
max_snr_db = 13  # Maximum SNR in dB
ber = []  # List to store BER for each SNR

# Loop over SNR values
for snr_db in range(max_snr_db):
    # Generate random binary data
    binary_data = np.random.randint(0, 2, (data_size, 4))  # 4 bits per symbol
    binary_to_decimal = np.array([8, 4, 2, 1])  # Binary to decimal mapping
    decimal_indices = np.dot(binary_data, binary_to_decimal)  # Convert to symbol indices

    # Map to QAM symbols
    tx_symbols = QAM_16[decimal_indices]

    # Add AWGN noise
    noise_std = 10 ** (-snr_db / 20)
    noise = (np.random.randn(data_size) + 1j * np.random.randn(data_size)) * noise_std / np.sqrt(2)
    rx_symbols = tx_symbols + noise

    # Demodulate symbols
    detected_symbols = np.argmin(np.abs(rx_symbols[:, None] - QAM_16[None, :]), axis=1)
    detected_bits = np.array([QAM_to_bit[idx] for idx in detected_symbols])

    # Calculate bit errors
    bit_errors = np.sum(binary_data != detected_bits)
    ber.append(bit_errors / (data_size * 4))  # Normalize by total transmitted bits

# Plot BER vs SNR
snr = np.arange(0, max_snr_db)

plt.semilogy(snr, ber)
plt.show()
