import numpy as np
from numpy import linalg as LA
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from pylab import *

def pca(X):

	# Normalize signal
	B2 = []
	for x in X:
		B2.append(x - np.ones_like(x)*np.mean(x))


	# Discard 25% of the points with the highest norm value
	alpha = 0.25
	norms = []
	for b in np.transpose(B2):
		norms.append(np.linalg.norm(b))
	ind = np.argsort(norms)
	nb_norms = np.ceil((1-alpha) * len(norms)).astype(int)
	ind = ind[nb_norms:]
	B = np.array(B2)[:,ind]

	# Compute PCA with the discarded points above
	# but use all the points when doing the projection later (B2.V)
	C = np.cov(B)		# Covariance
	lam, V = LA.eig(C)  # Eigenvalues, eigenvectors
	V = V.real
	lam = lam.real

	# not used, add threshold to keep only the eigenvectors with the highest eigenvalues
	ind = np.argsort(lam)[::-1]
	V = V[:,ind]

	# Project the original data on the new dimensions
	T = np.dot(np.transpose(B2),V)
	return T, B2, V

def butterworth_passband(lowcut, highcut, Fs, order=5):

	nyq = 0.5 * Fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a

def butterworth_passband_filter(signal, lowcut, highcut, Fs, order=5):
	b, a = butterworth_passband(lowcut, highcut, Fs)
	y = filtfilt(b, a, signal)
	return y

def signal_processing(y, x, new_x, lowcut, highcut, fps, Ns):

	# Interpolate y using x as reference
	t_func = interp1d(x, y)
	y = t_func(new_x)

	# Filter frequencies
	y = butterworth_passband_filter(y, lowcut, highcut, fps)

	# Power spectrum (only positive values)
	y = fft(y)
	y = 2/Ns * np.abs(y[:Ns//2])
	return y

def get_periodicity(power, freq):
	# Periodicity is defined as the percentage of power that the fundamental frequency (highest peak) and its first
	# harmonic (2 * fundamental frequency) occupy over the total signal power

	idx = np.argsort(power)[::-1]
	max = freq[idx[0]]								 # Highest peak (frequency)
	max_power = power[np.where(freq == max)][0]		 # Highest peak (power)
	first_power = power[np.where(freq == 2*max)][0]  # First harmonic of highest peak

	sum = np.sum(power)
	return (max_power+first_power)/sum	 # Percentage of highest peak and its first harmonic over all power spectrum


def process_tracker(y, fps, lowcut, highcut, Fs):

	# y = [nb_points x nb_frames]

	nb_frames = len(y[0])
	sec = nb_frames / fps  # Nb of frames per second
	time_axis = np.linspace(0, sec, nb_frames)

	# Fs = Desired sampling frequency (250 Hz)
	Ns = np.ceil(Fs*nb_frames/fps).astype(int)  # Nb of samples (for 250 Hz)
	new_time_axis = np.linspace(0, sec, Ns)
	freq = np.linspace(0, Fs/2, Ns/2)

	# Delete points where acquisition == 0 (means the point was lost)
	X = []
	for x in y:
		if len(np.where(x == 0)[0]) <= 1:
			X.append(x)

	# Compute PCA
	T, _, _ = pca(X)

	per = []
	spec = []
	for t in np.transpose(T):
		# Interpolate to 250 Hz, filter frequencies and compute fft
		y = signal_processing(t, time_axis, new_time_axis, lowcut, highcut, Fs, Ns)

		# Periodicity of power spectrum
		p = get_periodicity(y, freq)
		per.append(p)
		spec.append(y)

	# Get power spectrum with highest periodicity
	idx = np.argsort(per)[::-1][0]

	# Compute diff and find second peak of power spectrum (frequency corresponding to heart pulse)
	diff = spec[idx] - np.concatenate(([0],spec[idx][:-1]))
	diff = np.concatenate((diff[1:], [0]))
	bpm = 60 * freq[np.argsort(diff)[::-1][1] + 1]

	# Plot power spectrum of every component of the pca (same as number of originaly tracked points)
	# (but only one power spectrum (the one with the highest periodicity) is used)
	i=1
	for t in spec:
		plt.subplot(len(spec),1,i)
		plt.plot(60*freq[:100], t[:100])
		i+=1

	return bpm	# beats epr minute