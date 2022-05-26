import numpy as np
import matplotlib.pyplot as plt

#datadir = "C:\\Users\\tajdy\\Documents\\SLAC\\nexus_data\\2022-04-20\\"
#datadir="C:\\Users\\tajdy\\Documents\\SLAC\\nexus_data\\2022-05-25\\"
datadir="C:\\Users\\tajdy\\Documents\\SLAC\\nexus_data\\forTaj\\"
'''
linmag = np.load(f"{datadir}TTCW_scan_results_manypowers_overnight_Mag.npy")
linphase = np.load(f"{datadir}TTCW_scan_results_manypowers_overnight_Phase.npy")
bl_linmag = np.load(f"{datadir}TTCW_scan_baseline_manypowers_overnight_Mag.npy")
bl_linphase = np.load(f"{datadir}TTCW_scan_baseline_manypowers_overnight_Phase.npy")
'''
'''
linmag = np.load(f"{datadir}debug_Mag.npy")
linphase = np.load(f"{datadir}debug_Phase.npy")
bl_linmag = np.load(f"{datadir}debug_baseline_Mag.npy")
bl_linphase = np.load(f"{datadir}debug_baseline_Phase.npy")
'''

linmag = np.load(f"{datadir}qubit_1__Mag.npy")
linphase = np.load(f"{datadir}qubit_1__Phase.npy")
bl_linmag = np.load(f"{datadir}qubit_1__baseline_Mag.npy")
bl_linphase = np.load(f"{datadir}qubit_1__baseline_Phase.npy")


print(linmag.shape)
print(bl_linmag.shape)

# fq, pr, fr

# taken from the script
# TODO implement some kind of parameter saving like for the TWPA
VNA_fi = 5.82826
VNA_ff = 5.82848
aux_fi = 4.5
aux_ff = 4.8

dark = 5.8284046
bright = 5.827832

#frs = np.linspace(VNA_fi, VNA_ff, 31)
frs = np.linspace(VNA_fi, VNA_ff, 5)

dark_i = 2

ps = np.linspace(-75,-35,bl_linmag.shape[0])

'''
display = (linmag-bl_linmag)[:,:,dark_i]

extent = [ps[0], ps[-1], aux_ff, aux_fi]
extreme = np.max((np.max(display), -np.min(display)))

plt.xlabel("Readout Power at Feedthrough (dBm)")
plt.ylabel("Qubit Pump Frequency (GHz)")
plt.title(f"IFBW: 1 kHz, 100 avg count, \nReadout Freq: {np.round(frs[dark_i],6)} GHz (dark state), Qubit Power: -35 dBm")

plt.imshow(display, extent=extent, interpolation='none', aspect='auto', cmap="RdYlBu_r", vmin=-extreme, vmax=extreme)
cbar = plt.colorbar()
cbar.set_label("S43 Magnitude (linear), baseline subtracted")
plt.show()
'''

ind = 0

mag = np.sqrt(linmag**2+linphase**2)
phase = np.arctan2(linmag, linphase)
blmag = np.mean(np.sqrt(linmag**2+linphase**2)[:10], axis=0)
blphase = np.mean(np.arctan2(linmag, linphase)[:10], axis=0)
#blmag = np.sqrt(bl_linmag**2+bl_linphase**2)
#blphase = np.arctan2(bl_linmag, bl_linphase)

print(blmag[ind].shape)

plt.figure()
plt.title("bl subtracted")
plt.imshow(mag[:,ind]-blmag[ind],  aspect='auto', interpolation='none', cmap="RdYlBu_r")
plt.colorbar()
plt.figure()
plt.title("raw")
plt.imshow(mag[:,ind],  aspect='auto', interpolation='none', cmap="RdYlBu_r")
plt.colorbar()


for i in range(ps.size):
	if i == ind:
		plt.figure()
		plt.title(f"Baseline S43, Magnitude")
		plt.ylabel("S43 (linear [uV?])")
		plt.xlabel("Readout Frequency (GHz)")
		plt.plot(10*np.log10(blmag[i]))
	

'''
plt.figure()
plt.title(f"Baseline S43, Magnitude")
plt.ylabel("Readout Power (dBm)")
plt.xlabel("Readout Frequency (GHz)")
extent = [VNA_fi, VNA_ff, ps[-1], ps[0]]
for i in range(ps.size):
	bl_linmag[:i]*=1e3
plt.imshow(bl_linmag, extent=extent, interpolation='none', aspect='auto')
'''


'''

for i in range(ps.size):
	plt.figure()
	display = linmag[:,i] - bl_linmag[i]

	extent = [VNA_fi, VNA_ff, aux_ff, aux_fi]
	extreme = np.max((np.max(display), -np.min(display)))

	plt.xlabel("Readout Frequency (GHz)")
	plt.ylabel("Qubit Pump Frequency (GHz)")
	plt.title(f"IFBW: 1 kHz, 100 avg count, \npowers (at feedthrough): Readout: {np.round(ps[i],2)} dBm, Pump: -35 dBm")

	plt.plot(np.ones(2)*bright, (aux_fi, aux_ff), 'r--')
	plt.plot(np.ones(2)*dark, (aux_fi, aux_ff), 'b--')

	plt.imshow(display, extent=extent, interpolation='none', aspect='auto', cmap="RdYlBu_r", vmin=-extreme, vmax=extreme)
	cbar = plt.colorbar()
	cbar.set_label("S43 Magnitude (linear), baseline subtracted")
'''
plt.show()

exit()