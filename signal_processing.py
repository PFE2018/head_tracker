import numpy as np
from numpy import linalg as LA
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from pylab import *
import rospy
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2


class SIGNAL(object):

	def __init__(self, lowcut, highcut, Fs, wd, alpha):

		self.lowcut = lowcut
		self.highcut = highcut
		self.Fs = Fs
		self.wd = wd
		self.alpha = alpha

		self.points_timeserie = {
			'time': [],
			'values': [],
		}

	def process_tracker(self, msg):

		y = []
		# Extract list of xyz coordinates of point cloud
		for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
			y.append(p['y'])
		t = (msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)

		# Store  points
		self.points_timeserie['time'].append(t)
		self.points_timeserie['values'].append(y)
		print(str(t - self.points_timeserie['time'][0]) + 'seconds elapsed')

		if t - self.points_timeserie['time'][0] > self.wd:
			# Transfer to numpy array
			self.points_timeserie['time'] = np.asarray(self.points_timeserie['time'])
			self.points_timeserie['time'] = self.points_timeserie['time'] - self.points_timeserie['time'][0]
			self.points_timeserie['values'] = np.asarray(self.points_timeserie['values'])

			nb_frames = len(self.points_timeserie['values'])
			#sec = nb_frames / fps  # Nb of frames per second
			fps = nb_frames/self.wd
			time_axis = np.linspace(0, self.wd, nb_frames)

			# Fs = Desired sampling frequency (250 Hz)
			self.Ns = np.ceil(self.Fs * nb_frames / fps).astype(int)  # Nb of samples (for 250 Hz)
			new_time_axis = np.linspace(0, self.wd, self.Ns)
			self.freq = np.linspace(0, self.Fs/2, self.Ns/2)

			# Compute PCA
			T, _, _ = self.pca(self.points_timeserie['values'])

			periodicities = []
			spectrums = []
			for t in np.transpose(T):
				# Interpolate to 250 Hz, filter frequencies and compute fft
				y = self.signal_processing(t, time_axis, new_time_axis)

				# Periodicity of power spectrum
				p = self.get_periodicity(y)
				periodicities.append(p)
				spectrums.append(y)

			# Get power spectrum with highest periodicity
			idx = np.argsort(periodicities)[::-1][0]

			# Compute diff and find second peak of power spectrum (frequency corresponding to heart pulse)
			diff = spectrums[idx] - np.concatenate(([0], spectrums[idx][:-1]))
			diff = np.concatenate((diff[1:], [0]))
			bpm = 60 * self.freq[np.argsort(diff)[::-1][1] + 1]

			# Plot power spectrum of every component of the pca (same as number of originaly tracked points)
			# (but only one power spectrum (the one with the highest periodicity) is used)
			i = 1
			for t in spectrums:
				plt.subplot(len(spectrums), 1, i)
				plt.plot(60 * self.freq[:100], t[:100])
				i += 1
			plt.xlabel('BPM')
			plt.ylabel('Amplitude')

			return bpm  # beats per minute


	def pca(self, X):

		# Normalize signal
		B2 = []
		for x in X:
			B2.append(x - np.ones_like(x)*np.mean(x))


		# Discard 25% (alpha) of the points with the highest norm value
		norms = []
		for b in np.transpose(B2):
			norms.append(np.linalg.norm(b))
		ind = np.argsort(norms)
		nb_norms = np.ceil((1-self.alpha) * len(norms)).astype(int)
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

	def butterworth_passband(self, order=5):

		nyq = 0.5 * self.Fs
		low = self.lowcut / nyq
		high = self.highcut / nyq
		b, a = self.butter(order, [low, high], btype='band')
		return b, a

	def butterworth_passband_filter(self, signal, order=5):
		b, a = self.butterworth_passband()
		y = filtfilt(b, a, signal)
		return y

	def signal_processing(self, y, x, new_x):

		# Interpolate y using x as reference
		t_func = interp1d(x, y)
		y = t_func(new_x)

		# Filter frequencies
		y = self.butterworth_passband_filter(y)

		# Power spectrum (only positive values)
		y = fft(y)
		y = 2/self.Ns * np.abs(y[:self.Ns//2])
		return y

	def get_periodicity(self, power):
		# Periodicity is defined as the percentage of power that the fundamental frequency (highest peak) and its first
		# harmonic (2 * fundamental frequency) occupy over the total signal power

		idx = np.argsort(power)[::-1]
		max = self.freq[idx[0]]								 # Highest peak (frequency)
		max_power = power[np.where(self.freq == max)][0]		 # Highest peak (power)
		first_power = power[np.where(self.freq == 2*max)][0]  # First harmonic of highest peak

		sum = np.sum(power)
		return (max_power+first_power)/sum	 # Percentage of highest peak and its first harmonic over all power spectrum


if __name__ == '__main__':

	lowcut = 0.75		# Butterworth filter lowcut freq
	highcut = 5			# Butterworth filter highcut freq
	Fs = 250 			# interpolated frame rate
	wd = 300 			# seconds
	alpha = 0.25 		# % of points to discard
	head_tracker = SIGNAL(lowcut, highcut, Fs, wd, alpha)
	rospy.init_node('pca_head_tracker')
	rospy.Subscriber('/filtered_pcloud', PointCloud2, head_tracker.process_tracker)
	rospy.spin()