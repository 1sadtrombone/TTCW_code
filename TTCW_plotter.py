import numpy as np
import matplotlib.pyplot as plt

datadir = "C:\\Users\\tajdy\\Documents\\SLAC\\nexus_data\\2022-04-20"

linmag = np.load(f"{datadir}\\TTCW_scan_results_manypowers_overnight_Mag.npy")
linphase = np.load(f"{datadir}\\TTCW_scan_results_manypowers_overnight_Phase.npy")
bl_linmag = np.load(f"{datadir}\\TTCW_scan_baseline_manypowers_overnight_Mag.npy")
bl_linphase = np.load(f"{datadir}\\TTCW_scan_baseline_manypowers_overnight_Phase.npy")

# fq, pr, fr

# taken from the script
# TODO implement some kind of parameter saving like for the TWPA
VNA_fi = 5.82826
VNA_ff = 5.82848
aux_fi = 4.5
aux_ff = 4.8

dark = 5.8284046
bright = 5.827832

frs = np.linspace(VNA_fi, VNA_ff, 31)

dark_i = 20

ps = np.linspace(-55,-15,30) - 20

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

'''
extent = [VNA_fi, VNA_ff, aux_ff, aux_fi]
plt.imshow(np.log10(linmag), extent=extent,  aspect='auto', interpolation='none')

plt.figure()
'''
'''
for i in range(ps.size):
	plt.figure()
	plt.title(f"{ps[i]} Baseline S43, Magnitude")
	plt.ylabel("S43 (linear [uV?])")
	plt.xlabel("Readout Frequency (GHz)")
	plt.plot(np.linspace(VNA_fi,VNA_ff,bl_linmag.shape[1]), bl_linmag[i])
	

plt.figure()
plt.title(f"Baseline S43, Magnitude")
plt.ylabel("Readout Power (dBm)")
plt.xlabel("Readout Frequency (GHz)")
extent = [VNA_fi, VNA_ff, ps[-1], ps[0]]
for i in range(ps.size):
	bl_linmag[:i]*=1e3
plt.imshow(bl_linmag, extent=extent, interpolation='none', aspect='auto')

plt.show()

exit()
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
plt.show()
