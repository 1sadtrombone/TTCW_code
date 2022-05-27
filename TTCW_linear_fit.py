import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from TTCW_plotter import readData

def lorentzian(x, x0, a, gam):
    return a * gam**2 / (gam**2 + (x - x0)**2)

datadir = "C:\\Users\\tajdy\\Documents\\SLAC\\nexus_data\\2022-05-26\\"

fit_starts = [5, 5, 5, 5] # the power index where the resonance falls into the noise
fit_ends = [18, 16, 16, 16] # the power ind where the resonance escapes the range of qubit freqs

for q in range(4):
	if q == 2:
		continue

	tag = f"qubit_{q+1}_stark"

	linRe, linIm, bl_linRe, bl_linIm, prm = readData(datadir, tag)
	mag = np.sqrt(linRe**2+linIm**2)
	blmag = np.sqrt(bl_linRe**2+bl_linIm**2)

	ps_log = prm["prs"]
	ps_lin = 10**(np.array(ps_log)/10)

	fqs = prm["fqs"]

	fit_start = fit_starts[q]
	fit_end = fit_ends[q]  

	peak_fqs = np.zeros(fit_end-fit_start)
	peak_linps = np.zeros(fit_end-fit_start)
	hwhms = 0*peak_linps

	for i,j in enumerate(range(fit_start,fit_end)):

		data = mag[0,j,:,0]-blmag[0,j]

		center = np.argmax(data)
		peak_linps[i] = ps_lin[j]
		#peak_fqs[i] = fqs[center]

		# fitting lorentzians

		median = np.median(data)
		margin = 1*np.std(data)

		plt.plot(data)
		plt.show()

		minrange = np.where(data[:center] < median + margin)[0][-1]
		maxrange = np.where(data[center:] < median + margin)[0][0] + center

		if maxrange - minrange < 3:
			minrange -= 3 - (maxrange - minrange)

		print(i+fit_start)

		minf = fqs[minrange]
		maxf = fqs[maxrange]

		popt, cov = curve_fit(lorentzian, fqs[minrange:maxrange], data[minrange:maxrange], p0=[fqs[center], 1e-3, 1e-3], bounds=([minf,1e-4,1e-4],[maxf,1e-2,5e-2]))
		
		peak_fqs[i] = popt[0]
		hwhms[i] = popt[2]

		'''
		plt.figure(figsize=(10,5))
		plt.plot(fqs,data)
		plt.plot(fqs,np.ones_like(data)*median+margin, 'k--')
		plt.plot((fqs[minrange],fqs[maxrange]),(data[minrange], data[maxrange]),'r.')
		plt.plot(fqs, lorentzian(fqs, *popt))
		plt.plot(fqs[center], data[center], 'b.')
		plt.xlabel("Qubit Frequency (GHz)")
		plt.ylabel("S43")
		plt.title(f"Readout Power: {peak_linps[i]} mW")
		plt.show()
		'''
	

	# N_gamma - power calibration

	resonator_ground_f = 5.8284046e9 # Hz
	resonator_excited_f = 5.827832e9 
	resonator_highp_f = 5.8277055e9 

	peak_fqs_Hz = peak_fqs*1e9 # EVERYTHING must be in Hz

	params = np.polyfit(peak_linps, peak_fqs_Hz, deg=1)
	slope, intercept = params

	chi = (resonator_ground_f - resonator_excited_f) / 2 # Hz

	N_gamma_slope = slope / (-2*(chi)) # this is photons/mW

	# output some stuff

	print(f"chi: {chi}")
	print(f"photons at max. power: {(peak_linps[-1])*N_gamma_slope}")
	print(f"photons per mW: {N_gamma_slope}")
	print(f"mW for one photon: {1/N_gamma_slope}")
	print(f"dBm for one photon: {10*np.log10(1/N_gamma_slope)}")

	# Getting Q

	fwhms = 2*hwhms
	Qs = peak_fqs/fwhms

	# display

	frs = prm["frs"]

	plt.figure()
	display = mag[0,:,:,0].T-blmag[0]

	extent = [ps_log[0], ps_log[-1], np.max(prm["fqs"]), np.min(prm["fqs"])]
	extreme = np.max((np.max(display), -np.min(display)))

	plt.xlabel("Readout Power at Feedthrough (dBm)")
	plt.ylabel("Qubit Pump Frequency (GHz)")
	plt.title(f"IFBW: 1 kHz, 100 avg count, \nReadout Freq: {np.round(frs[0],6)} GHz (dark state), Qubit Power: -35 dBm")

	plt.imshow(display, extent=extent, interpolation='none', aspect='auto', cmap="RdYlBu_r", vmin=-extreme, vmax=extreme)
	cbar = plt.colorbar()
	cbar.set_label("S43 Magnitude (linear), baseline subtracted")
	plt.plot(10*np.log10(peak_linps), peak_fqs, 'k.')

	plt.figure()

	plt.semilogx(N_gamma_slope*peak_linps, peak_fqs_Hz, 'k.')
	plt.semilogx(N_gamma_slope*peak_linps, params[0]*peak_linps+params[1], 'r--', label=f"Slope: {np.round(params[0],5)}\nIntercept: {np.round(params[1],6)}")
	plt.title("AC Stark Shift")
	plt.xlabel("Number of Photons")
	plt.ylabel("Qubit Pump Frequency (GHz)")
	plt.legend()

	plt.figure()
	plt.semilogx(N_gamma_slope*peak_linps, fwhms, 'ko-')
	plt.title("Linewidth Dependence")
	plt.xlabel("Number of Photons")
	plt.ylabel("Lorentzian Linewidth [FWHM] (GHz)")

	plt.figure()
	plt.semilogx(N_gamma_slope*peak_linps, Qs, 'ko-')
	plt.title("Quality Factor Dependence")
	plt.xlabel("Number of Photons")
	plt.ylabel("Q")

	plt.show()