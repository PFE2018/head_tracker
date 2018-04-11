import numpy as np
from numpy import linalg as LA
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from pylab import *
import argparse
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2


class SIGNAL:

    def __init__(self, args):

        self.args = args
        self.Ns = 0
        self.freq = None

    def main_loop(self, msg):

        tic = msg.data['tic']
        toc = msg.data['toc']

        if toc - tic >= self.args.process_time:
            tracked_points = msg.data['tracked_points']
            fps = msg.data['fps']

            nb_frames = len(tracked_points)
            wd = nb_frames / fps
            time_axis = np.linspace(0, wd, nb_frames)

            self.Ns = np.ceil(self.args.Fs * nb_frames / fps).astype(int)  # Nb of samples (for 250 Hz)
            new_time_axis = np.linspace(0, wd, self.Ns)
            self.freq = np.linspace(0, self.args.Fs / 2, self.Ns / 2)

            # Compute PCA
            T, _, _ = self.pca(tracked_points)

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

            print(str(bpm) + ' beats per min')  # beats per minute

    def pca(self, X):

        # Normalize signal
        B2 = []
        for x in X:
            B2.append(x - np.ones_like(x) * np.mean(x))

        # Discard 25% (alpha) of the points with the highest norm value
        norms = []
        for b in np.transpose(B2):
            norms.append(np.linalg.norm(b))
        ind = np.argsort(norms)
        nb_norms = np.ceil((1 - self.args.alpha) * len(norms)).astype(int)
        ind = ind[nb_norms:]
        B = np.array(B2)[:, ind]

        # Compute PCA with the discarded points above
        # but use all the points when doing the projection later (B2.V)
        C = np.cov(B)  # Covariance
        lam, V = LA.eig(C)  # Eigenvalues, eigenvectors
        V = V.real
        lam = lam.real

        # not used, add threshold to keep only the eigenvectors with the highest eigenvalues
        ind = np.argsort(lam)[::-1]
        V = V[:, ind]

        # Project the original data on the new dimensions
        T = np.dot(np.transpose(B2), V)
        return T, B2, V

    def butterworth_passband(self, order=5):

        nyq = 0.5 * self.args.Fs
        low = self.args.lowcut / nyq
        high = self.args.highcut / nyq
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
        y = 2 / self.Ns * np.abs(y[:self.Ns // 2])
        return y

    def get_periodicity(self, power):
        # Periodicity is defined as the percentage of power that the fundamental frequency (highest peak) and its first
        # harmonic (2 * fundamental frequency) occupy over the total signal power

        idx = np.argsort(power)[::-1]
        max = self.freq[idx[0]]  # Highest peak (frequency)
        max_power = power[np.where(self.freq == max)][0]  # Highest peak (power)
        first_power = power[np.where(self.freq == 2 * max)][0]  # First harmonic of highest peak

        sum = np.sum(power)
        return (
                       max_power + first_power) / sum  # Percentage of highest peak and its first harmonic over all power spectrum


# if __name__ == '__main__':
#     # ==================================================================================================================
#     parser = argparse.ArgumentParser(
#         description='Signal Processing of facial tracked points (head_tracker Subscriber Node)')
#     parser.add_argument('--process_time', type=int, default=20, help='Time of acquisition')
#     parser.add_argument('--lowcut', type=int, default=0.75, help='Lowcut frequency for Butterworth filter')
#     parser.add_argument('--highcut', type=int, default=5, help='Highcut frequency for Butterworth filter')
#     parser.add_argument('--Fs', type=int, default=250, help='Interpolated frame rate (Hz)')
#     parser.add_argument('--alpha', type=int, default=0.25, help='% of points to discard')
#
#     args = parser.parse_args()
#     # ==================================================================================================================
#
#     head_tracker = SIGNAL(args)
    # rospy.init_node('head_tracker')
    #
    # # TODO: Change Subscriber
    # rospy.Subscriber('/filtered_pcloud', String, head_tracker.main_loop)
    # rospy.spin()
