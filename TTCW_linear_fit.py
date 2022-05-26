import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def lorentzian(x, x0, a, gam):
    return a * gam**2 / (gam**2 + (x - x0)**2)


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
fqs = np.linspace(aux_fi, aux_ff, linmag.shape[0]) # GHz

dark_i = 20

ps_log = np.linspace(-55,-15,30) - 20
ps_lin = 10**(ps_log/10) # mW

fit_start = 8 # power index where the peak fades under noise
fit_end = 23 # power index where the peak gets driven normal

# Lorenztian fits

peak_fqs = np.zeros(fit_end-fit_start)
peak_linps = np.zeros(fit_end-fit_start)
hwhms = 0*peak_linps

for i,j in enumerate(range(fit_start,fit_end)):

	data = linmag[:,j,dark_i]-bl_linmag[dark_i,j]

	center = np.argmax(data)
	peak_linps[i] = ps_lin[j]
	#peak_fqs[i] = fqs[center]

	# fitting lorentzians

	median = np.median(data)
	margin = 1*np.std(data)

	minrange = np.where(data[:center] < median + margin)[0][-1]
	maxrange = np.where(data[center:] < median + margin)[0][0] + center

	if maxrange - minrange < 3:
		minrange -= 3 - (maxrange - minrange)

	print(i)

	minf = fqs[minrange]
	maxf = fqs[maxrange]

	popt, cov = curve_fit(lorentzian, fqs[minrange:maxrange], data[minrange:maxrange], p0=[fqs[center], 1e-3, 1e-3], bounds=([minf,1e-4,1e-4],[maxf,1e-2,5e-2]))
	
	peak_fqs[i] = popt[0]
	hwhms[i] = popt[2]

	# get Q eventually

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

plt.figure()
display = (linmag-bl_linmag)[:,:,dark_i]

extent = [ps_log[0], ps_log[-1], aux_ff, aux_fi]
extreme = np.max((np.max(display), -np.min(display)))

plt.xlabel("Readout Power at Feedthrough (dBm)")
plt.ylabel("Qubit Pump Frequency (GHz)")
plt.title(f"IFBW: 1 kHz, 100 avg count, \nReadout Freq: {np.round(frs[dark_i],6)} GHz (dark state), Qubit Power: -35 dBm")

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