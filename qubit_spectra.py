import numpy as np
import matplotlib.pyplot as plt

#datadir = "C:\\Users\\tajdy\\Documents\\SLAC\\nexus_data\\2022-04-20\\"
datadir="C:\\Users\\tajdy\\Documents\\SLAC\\nexus_data\\forTaj\\"

freqstrs = ["5827500000.0","5958800000.0","6074500000.0","6187500000.0"]
qubitnumber = [2, 4, 3, 1]

for i in range(4):

	# for each qubit
	print(i)
	lowp = np.load(f"{datadir}qubit_{i+1}_cavity_spectroscopy_power_m60_freq_{freqstrs[i]}_pm3000000.0.npy")
	hip = np.load(f"{datadir}qubit_{i+1}_cavity_spectroscopy_power_m20_freq_{freqstrs[i]}_pm3000000.0.npy")

	freqs = np.linspace(float(freqstrs[i]) - 3000000.0, float(freqstrs[i]) + 3000000.0, lowp.size)

	plt.figure()
	plt.title(f"Qubit {qubitnumber[i]} (Numbering consistent with previous slides)")
	plt.plot(freqs, lowp, label="Readout -75 dBm")
	plt.plot(freqs, hip, label="Readout -30 dBm")
	plt.xlabel("Readout Frequency (Hz)")
	plt.ylabel("S43 log mag")
	plt.legend()

plt.show()