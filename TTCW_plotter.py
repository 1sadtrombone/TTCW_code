import numpy as np
import matplotlib.pyplot as plt

datadir = "C:\\Users\\tajdy\\Documents\\SLAC\\nexus_data\\2022-04-14"

linmag = np.load(f"{datadir}\\TTCW_scan_results_Mag.npy")
linphase = np.load(f"{datadir}\\TTCW_scan_results_Phase.npy")
bl_linmag = np.load(f"{datadir}\\TTCW_scan_baseline_Mag.npy")
bl_linphase = np.load(f"{datadir}\\TTCW_scan_baseline_Phase.npy")


# taken from the script
# TODO implement some kind of parameter saving like for the TWPA
VNA_fi = 5.82645
VNA_ff = 5.82945
aux_fi = 4.5
aux_ff = 4.8

dark = 5.8284046
bright = 5.827832

plt.plot(np.linspace(VNA_fi,VNA_ff,bl_linmag.shape[0]),bl_linmag)
plt.show()

#ps = np.linspace(-55,-15,13)

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
'''

for i in range(1):
	plt.figure()
	display = linmag[:,i+12]

	extent = [VNA_fi, VNA_ff, aux_ff, aux_fi]
	extreme = np.max((np.max(display), -np.min(display)))

	plt.xlabel("Readout Frequency (GHz)")
	plt.ylabel("Qubit Pump Frequency (GHz)")
	plt.title(f"IFBW: 1 kHz, 100 avg count, \npowers (at feedthrough): Readout: {ps[i]-20} dBm, Pump: -35 dBm")

	plt.plot(np.ones(2)*bright, (aux_fi, aux_ff), 'r--')
	plt.plot(np.ones(2)*dark, (aux_fi, aux_ff), 'b--')

	plt.imshow(display, extent=extent, interpolation='none', aspect='auto', cmap="RdYlBu_r", vmin=-extreme, vmax=extreme)
	cbar = plt.colorbar()
	cbar.set_label("S43 Phase (linear), baseline subtracted")
plt.show()
'''