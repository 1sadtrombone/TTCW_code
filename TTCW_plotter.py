import numpy as np
import matplotlib.pyplot as plt

import json

def readData(datadir, tag, magphase=False):
	'''
	Read data and metadata json in directory datadir with prefix tag.
	'''

	# before I thought polar mode returned mag/phase, so that's how I named it
	# but it's always re/im

	if magphase:
		linRe = np.load(f"{datadir}{tag}_Mag.npy")
		linIm = np.load(f"{datadir}{tag}_Phase.npy")
		bl_linRe = np.load(f"{datadir}{tag}_baseline_Mag.npy")
		bl_linIm = np.load(f"{datadir}{tag}_baseline_Phase.npy")
		
	else:
		linRe = np.load(f"{datadir}{tag}_Re.npy")
		linIm = np.load(f"{datadir}{tag}_Im.npy")
		bl_linRe = np.load(f"{datadir}{tag}_baseline_Re.npy")
		bl_linIm = np.load(f"{datadir}{tag}_baseline_Im.npy")

	with open(f"{datadir}{tag}_metadata.json") as paramfile:
		prm = json.load(paramfile)

	return linRe, linIm, bl_linRe, bl_linIm, prm

	# fr, pr, fq, pq

def plotTwoTone(datadir, tag, magphase=False):
	'''
	Expecting the data it reads to be a full data cube, shape fr, pr, fq, pq
	Plots the full cube, making a colormap of fr and fq for each combination of reaodut and qubit powers
	'''

	linRe, linIm, bl_linRe, bl_linIm, prm = readData(datadir, tag, magphase)
	mag = np.sqrt(linRe**2+linIm**2)
	blmag = np.sqrt(bl_linRe**2+bl_linIm**2)

	fri = np.min(prm["frs"])*1e-9
	frf = np.max(prm["frs"])*1e-9
	fqi = np.min(prm["fqs"])*1e-9
	fqf = np.max(prm["fqs"])*1e-9

	for i,pr in enumerate(prm["prs"]):
		for j,pq in enumerate(prm["pqs"]):
			display = mag[:,i,:,j].T-blmag[:,i]
			extent = [frf, fri, fqi, fqf]
			extreme = np.max((np.max(display), -np.min(display)))
			plt.figure()
			plt.title(f"IFBW: {prm['IFBW']}, {prm['avg_cnt']} avg count, \nReadout Power: {pr-prm['readout_atten']} dBm, Qubit Power: {pq-prm['aux_atten']} dBm")
			plt.imshow(display, extent=extent, interpolation='none', aspect='auto', cmap="RdYlBu_r", vmin=-extreme, vmax=extreme)
			cbar = plt.colorbar()
			cbar.set_label("S43 Magnitude (linear), baseline subtracted")
			plt.ylabel("Qubit Pump Frequency (GHz)")
			plt.xlabel("Readout Frequency (GHz)")

def plotStarkShift(datadir, tag, magphase=False):
	'''
	Expecting to read data with a single qubit power and readout frequency. 
	Plots qubit frequency vs readout power. 
	'''

	linRe, linIm, bl_linRe, bl_linIm, prm = readData(datadir, tag, magphase)
	mag = np.sqrt(linRe**2+linIm**2)
	blmag = np.sqrt(bl_linRe**2+bl_linIm**2)

	pri = np.min(prm["prs"]) - prm["readout_atten"]
	prf = np.max(prm["prs"]) - prm["readout_atten"]
	fqi = np.min(prm["fqs"])*1e-9
	fqf = np.max(prm["fqs"])*1e-9
	fr = prm["frs"][0]*1e-9

	display = mag[0,:,:,0].T-blmag[0,:]

	extent = [pri, prf, fqi, fqf]
	extreme = np.max((np.max(display), -np.min(display)))

	plt.figure()
	plt.title(f"IFBW: {prm['IFBW']}, {prm['avg_cnt']} avg count, \nReadout Frequency: {fr} GHz, Qubit Power: {prm['pqs'][0]-prm['aux_atten']} dBm")
	plt.imshow(display, extent=extent, interpolation='none', aspect='auto', cmap="RdYlBu_r", vmin=-extreme, vmax=extreme)
	cbar = plt.colorbar()
	cbar.set_label("S43 Magnitude (linear), baseline subtracted")
	plt.ylabel("Qubit Pump Frequency (GHz)")
	plt.xlabel("Readout Power (dBm)")

if __name__=="__main__":

	datadir="C:\\Users\\tajdy\\Documents\\SLAC\\nexus_data\\2022-05-26\\"
	for i in range(4):

		#tag = f"qubit_{i+1}_pqs_scan"
		#plotTwoTone(datadir, tag)

		tag = f"qubit_{i+1}_stark"
		plotStarkShift(datadir, tag)

	plt.show()