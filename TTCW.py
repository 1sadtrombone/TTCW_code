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

def VNAPowerS21(vna, fs, powers, average_count):
	"""
	Scan S21 (or in our case actually S43) for several readout powers
	and plot the resulting 2D heatmap.
	"""

	vna.setFormat('mlog')
	vna.setNPoints(average_count)
	vna.setRFOn()

	data = np.zeros((powers.size,fs.size))

	print('working')
	for i,p in enumerate(powers):
		print(p)
		vna.setPower(p)
		for j,f in enumerate(fs):
			vna.setStartFreq(f)
			vna.setStopFreq(f)
			_,r,_ = vna.getData()
			data[i,j] = np.mean(r)

	plt.figure()
	extent = [fs[0], fs[-1], powers[-1]-10, powers[0]-10] # 10 dB on atten
	plt.imshow(data, extent=extent, aspect='auto', interpolation='none')

	vna.setRFOff()
	vna.setIntTrigger()

	return data

def AveragedS43(vna, fs, average_count):

	vna.setFormat('mlog')
	vna.setNPoints(average_count)
	vna.setRFOn()

	data = np.zeros((fs.size))

	for i,f in enumerate(fs):
		print(f"freq {i+1}/{fs.size}")
		vna.setStartFreq(f)
		vna.setStopFreq(f)
		_,r,_ = vna.getData()
		data[i] = np.mean(r)

	vna.setRFOff()

	return data


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

def TwoTone(readout_fs, readout_powers, aux_fs, aux_powers, avg_cnt, baseline_avg_cnt, IFBW, twpa_freq, twpa_power, twpa_atten, data_dir, tag, aux_port=2, aux_atten=10, readout_atten=10):
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
	twpa.turnOn()
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

	vnaAux.enable()

	dataRe = np.zeros((aux_fs.size,readout_powers.size,readout_fs.size))
	dataIm = np.zeros((aux_fs.size,readout_powers.size,readout_fs.size))

	baselineRe = np.zeros((readout_powers.size,readout_fs.size))
	baselineIm = np.zeros((readout_powers.size,readout_fs.size))	

	for h,p in enumerate(readout_powers):
		vna.setPower(p)

		vnaAux.enable()
		vna.setNPoints(avg_cnt)

		for g,pq in enumerate(aux_powers):
			vnaAux.setPower(pq)

			for i,aux_f in enumerate(aux_fs):
				vnaAux.setCWFreq(aux_f)

				for j,read_f in enumerate(readout_fs):
					print(f"Readout power {h+1}/{readout_powers.size}, aux power {g+1}/{aux_powers.size}, pump freq {i+1}/{aux_fs.size}, readout freq {j+1}/{readout_fs.size} | pq: {pq}, fq: {aux_f}, pr: {p}, fr: {read_f}")
					vna.setStartFreq(read_f)
					vna.setStopFreq(read_f)

					f,r,im = vna.getData()
					dataRe[i,h,j] += np.mean(r)
					dataIm[i,h,j] += np.mean(im)

		vnaAux.disable()
		vna.setNPoints(baseline_avg_cnt)

		for j,read_f in enumerate(readout_fs):
			print(f"Baseline for readout power {h+1}/{readout_powers.size} readout freq {j+1}/{readout_fs.size} | Readout power: {p}, Readout freq: {read_f}")
			vna.setStartFreq(read_f)
			vna.setStopFreq(read_f)
			
			f,r,im = vna.getData()
			baselineRe[h,j] += np.mean(r)
			baselineIm[h,j] += np.mean(im)

	# TODO make saving nice and save params too like the TWPA stuff
	np.save(f"{data_dir}\\{tag}_Mag", dataRe)
	np.save(f"{data_dir}\\{tag}_Phase", dataIm)
	np.save(f"{data_dir}\\{tag}_baseline_Mag", baselineRe)
	np.save(f"{data_dir}\\{tag}_baseline_Phase", baselineIm)
	print("Data saved.")

	twpa.turnOff()
	twpa.disconnectAll()
	four_port.set(1,95)
	four_port.set(2,95)
	vna.setRFOff()


	vna.setIntTrigger()

	#extent = [aux_fi, aux_ff,readout_powers[-1], readout_powers[0]]

	print("Run finished successfully.")
	

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


if __name__ == "__main__":

	data_dir="C:\\Users\\nexus\\Desktop\\Share\\Share\\Data\\2022-05-20"

	pqi = -45
	pqf = -25
	pqn = 3
	pqs = np.linspace(pqi,pqf,pqn)

	pri = -40
	prf = -40
	prn = 1
	prs = np.linspace(pri,prf,prn)

	#TwoTone(readout_fs, readout_powers, aux_fs, aux_powers, avg_cnt, baseline_avg_cnt, IFBW, twpa_freq, twpa_power, twpa_atten, data_dir, tag, aux_port=2, aux_atten=10, readout_atten=10)

	resonator_fs = [5.8275e9, 5.9588e9, 6.0745e9, 6.1875e9]
	ground_states = [5.828473e9, 5.95938e9, 6.075066e9, 6.188073e9]
	readout_span = 3e6
	readout_nf = 51
	qubit_transition_freqs = [4.65e9, 4.65e9, 4.8e9, 4.8e9]
	qubit_span = 30e6
	qubit_nf = 151

	hifi_S43_nf = 501
	hifi_S43_navg = 1000
	hifi_powers = np.array([0,-40])

	# high fidelity S43 of qubits at high and low readout powers

	vna = VNA.VNA()

	for j,p in enumerate(hifi_powers):
		vna.setPower(p)
		print(f"power {p}"+" -"*30)
		for i in range(len(resonator_fs)):
			print(f"qubit {i+1}"+" -"*30)

			tag = f"qubit_{i+1}_"
			fs = np.linspace(resonator_fs[i]-readout_span, resonator_fs[i]+readout_span, hifi_S43_nf)
			np.save(f"{data_dir}/{tag}cavity_spectroscopy_power_m{-p+20}_freq_{resonator_fs[i]}_pm{readout_span}.npy",AveragedS43(vna, fs, hifi_S43_navg))

	# two tone scans

	for i in range(len(resonator_fs)):
		print(f"qubit {i+1}"+" -"*30)

		tag = f"qubit_{i+1}_"
		
		frs = np.linspace(ground_states[i]-readout_span, ground_states[i]+readout_span, readout_nf)
		fqs = np.linspace(qubit_transition_freqs[i]-qubit_span, qubit_transition_freqs[i]+qubit_span, qubit_nf)
		TwoTone(frs, prs, fqs, pqs, 100, 1000, 1e3, 9.4e9, 0, 8, data_dir, tag, aux_port=2, aux_atten=10, readout_atten=20)

	plt.show()