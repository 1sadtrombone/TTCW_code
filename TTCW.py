import sys
sys.path.append('C:\\Users\\nexus\\Desktop\\Share\\Share\\CodeSync\\Device_control')
#sys.path.append('Z:\\CodeSync\\Device_control')
#sys.path.append('C:\\Users\\tajdy\\Documents\\SLAC\\TWPA_code\\Device_control')

import Labber
import TWPA
import Attenuator
import VNA
import numpy as np
import matplotlib.pyplot as plt

# before running any of these things, ensure HEMT and TWPA are on
# as of writing, best TWPA params are 9.2G and 0 dBm power pump, 7 dB on the attenuator

# all of this was done with VNA atten 10 dB

def VNAPowerS21(center_f, f_width, npts, powers, average_count):
	"""
	Scan S21 (or in our case actually S43) for several readout powers
	and plot the resulting 2D heatmap.
	"""

	vna.setStartFreq(center_f-f_width)
	vna.setStopFreq(center_f+f_width)
	vna.setNPoints(npts)

	vna.setFormat('mlog')

	vna.setRFOn()

	data = np.zeros((powers.size,npts))

	print('working')
	for i,p in enumerate(powers):
		print(p)
		vna.setPower(p)
		for j in range(average_count):

			f,r,_ = vna.getData()
			data[i] += r
		data[i] /= average_count


	extent = [fc-fw, fc+fw, powers[0], powers[-1]]
	plt.imshow(data, extent=extent, aspect='auto', interpolation='none')

	plt.show()

	vna.setRFOff()
	vna.setIntTrigger()

def ScanAuxAtOneReadoutFreq(readout_f, avg_cnt, readout_power, aux_power, aux_fi, aux_ff, aux_npts, aux_port=2):
	"""

	"""

	aux_fs = np.linspace(aux_fi, aux_ff, aux_npts)

	# set up VNA 

	vna = VNA.VNA()
	vna.setStartFreq(readout_f)
	vna.setStopFreq(readout_f)
	vna.setNPoints(avg_cnt)

	vna.setPower(readout_power)

	vna.setFormat('mlog')

	vna.setRFOn()

	# set up aux source on VNA

	vnaAux = VNA.VNAAux()
	vnaAux.setPort(aux_port)
	vnaAux.setPower(aux_power)

	vnaAux.enable()

	data = np.zeros(aux_fs.size)

	for i,aux_f in enumerate(aux_fs):
		print(aux_f)
		vnaAux.setCWFreq(aux_f)
		f,r,_ = vna.getData()
		data[i] = np.mean(r)

	vna.setRFOff()
	vnaAux.disable()

	vna.setIntTrigger()

	plt.plot(aux_fs,data[:,0].T)
	plt.show()

def TwoTone(readout_fs, readout_powers, aux_fs, avg_cnt, baseline_avg_cnt, IFBW, aux_power, twpa_freq, twpa_power, twpa_atten, aux_port=2, aux_atten=10, readout_atten=10):
	"""
	The measurement of S43 as a func of readout frequency, qubit freq, and readout power.
	some sample values below.
	
	twpa_freq = 9.2e9
	twpa_power = 0
	twpa_atten = 7

	VNA_fi = 5.82645e9
	VNA_ff = 5.82945e9
	npts = 501

	IFBW = 1e3

	readout_power = -50 # dBm
	avg_cnt = 100
	baseline_avg_cnt = 1000

	aux_power = -25
	aux_fi = 4.6e9
	aux_ff = 4.7e9
	aux_npts = 501
	"""

	vna_format = 'polar'

	# set up TWPA
	client = Labber.connectToServer('localhost', timeout=30)

	twpa = TWPA.TWPA(client)

	twpa.connectAll()
	twpa.setPump(twpa_freq,twpa_power,twpa_atten)
	print("TWPA set to:")
	print(twpa.getPump())

	# set up attens

	four_port = Attenuator.Attenuator('COM4', verbose=False)
	four_port.set(1,aux_atten)
	four_port.set(2,readout_atten)

	# set up VNA 

	vna = VNA.VNA()

	vna.setIFBW(IFBW)

	vna.setFormat(vna_format)

	vna.setRFOn()

	# set up aux source on VNA

	vnaAux = VNA.VNAAux()
	vnaAux.setPort(aux_port)
	vnaAux.setPower(aux_power)

	vnaAux.enable()

	dataRe = np.zeros((aux_fs.size,readout_powers.size,readout_fs.size))
	dataIm = np.zeros((aux_fs.size,readout_powers.size,readout_fs.size))

	baselineRe = np.zeros((readout_powers.size,readout_fs.size))
	baselineIm = np.zeros((readout_powers.size,readout_fs.size))

	for h,p in enumerate(readout_powers):
		vna.setPower(p)

		vnaAux.enable()
		vna.setNPoints(avg_cnt)

		for i,aux_f in enumerate(aux_fs):
			vnaAux.setCWFreq(aux_f)

			for j,read_f in enumerate(readout_fs):
				print(f"(Superloop {h+1} of {readout_powers.size}) Qubit freq: {aux_f}, Readout power: {p}, Readout freq: {read_f}")
				vna.setStartFreq(read_f)
				vna.setStopFreq(read_f)

				f,r,im = vna.getData()
				dataRe[i,h,j] += np.mean(r)
				dataIm[i,h,j] += np.mean(im)

		vnaAux.disable()
		vna.setNPoints(baseline_avg_cnt)

		print("Taking Baselines...")
		for j,read_f in enumerate(readout_fs):
			print(f"(Baselines for superloop {h+1} of {readout_powers.size}) Readout power: {p}, Readout freq: {read_f}")
			vna.setStartFreq(read_f)
			vna.setStopFreq(read_f)
			
			f,r,im = vna.getData()
			baselineRe[h,j] += np.mean(r)
			baselineIm[h,j] += np.mean(im)

	twpa.disconnectAll()
	four_port.set(1,95)
	four_port.set(2,95)
	vna.setRFOff()


	vna.setIntTrigger()

	#extent = [aux_fi, aux_ff,readout_powers[-1], readout_powers[0]]

	# TODO make saving nice and save params too like the TWPA stuff
	np.save("TTCW_scan_results_manypowers_Mag", dataRe)
	np.save("TTCW_scan_results_manypowers_Phase", dataIm)
	np.save("TTCW_scan_baseline_manypowers_Mag", baselineRe)
	np.save("TTCW_scan_baseline_manypowers_Phase", baselineIm)

	print("Data saved. Run finished successfully.")
	

	#plt.imshow(dataRe[:,:,0]-baselineRe[:,0], extent=extent, interpolation='none', aspect='auto')
	#plt.show()

def TwoToneBaselineOnly(VNA_fi, VNA_ff, npts, IFBW, readout_power, baseline_avg_cnt, twpa_freq, twpa_power, twpa_atten, aux_atten=10, readout_atten=10):
	"""
	For if you need to retake only the baseline measurement for some reason.
	This only exists because I needed it once and decided not to delete it.
	Below are some sample values.
	
	twpa_freq=9.2e9
	twpa_power = 0
	twpa_atten = 7

	vna_format = 'polar'

	VNA_fi = 5.82645e9
	VNA_ff = 5.82945e9
	npts = 501

	IFBW = 1e3

	readout_power = -50 # dBm
	baseline_avg_cnt = 1000
	"""

	# set up TWPA
	client = Labber.connectToServer('localhost', timeout=30)

	twpa = TWPA.TWPA(client)

	twpa.connectAll()
	twpa.setPump(twpa_freq,twpa_power,twpa_atten)
	print("TWPA set to:")
	print(twpa.getPump())

	# set up attens

	four_port = Attenuator.Attenuator('COM4', verbose=False)
	four_port.set(1,aux_atten)
	four_port.set(2,readout_atten)

	# set up VNA 

	vna = VNA.VNA()
	vna.setStartFreq(VNA_fi)
	vna.setStopFreq(VNA_ff)
	vna.setNPoints(npts)

	vna.setPower(readout_power)
	vna.setIFBW(IFBW)

	vna.setFormat(vna_format)

	vna.setRFOn()

	# set up aux source on VNA

	vnaAux = VNA.VNAAux()
	vnaAux.setPort(2)
	vnaAux.disable()

	baselineRe = np.zeros(npts)
	baselineIm = np.zeros(npts)


	for j in range(baseline_avg_cnt):
		if j%(baseline_avg_cnt//10) == 0:
			print(j)
		f,r,im = vna.getData()
		baselineRe += r
		baselineIm += im

	baselineRe /= baseline_avg_cnt
	baselineIm /= baseline_avg_cnt

	np.save("TTCW_scan_real_baseline_Mag", baselineRe)
	np.save("TTCW_scan_real_baseline_Phase", baselineIm)

fqi = 4.5e9
fqf = 4.8e9
fqn = 91
fqs = np.linspace(fqi, fqf, fqn)

pi = -55
pf = -15
pn = 13
ps = np.linspace(pi,pf,pn)

fri = 5.82826e9
frf = 5.82848e9
frn = 7
frs = np.linspace(fri, frf, frn)

#TwoTone(readout_fs, readout_powers, aux_fs, avg_cnt, baseline_avg_cnt, IFBW, aux_power, twpa_freq, twpa_power, twpa_atten, aux_port=2, aux_atten=10, readout_atten=10)
TwoTone(frs, ps, fqs, 100, 1000, 1e3, -25, 9.2e9, 0, 7, aux_port=2, aux_atten=10, readout_atten=20)