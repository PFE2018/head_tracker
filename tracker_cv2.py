import numpy as np
import cv2
import dlib
from tools import rect_to_boundingbox
from acp import *
from imutils.video import FPS
import time
from signal_processing_cv2 import *

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
					  qualityLevel=0.3,
					  minDistance=7,
					  blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
				 maxLevel=2,
				 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

class TRACKER:

	def __init__(self, signal):

		self.tracking = False
		self.detector = dlib.get_frontal_face_detector()
		self.fps = FPS().start()
		self.y = []
		self.tic = time.time()
		self.toc = time.time()
		self.signal = signal

	def concat_points(self, points, new_points, idx_goodpoints):

		try:
			n = points.shape[1]
		except(IndexError):
			points = np.reshape(points, (len(points), 1))

		points = np.concatenate((points, np.zeros((len(points), 1))), axis=1)

		j = 0
		for i in range(len(points[:,-1])):

			if points[i,-2:-1] != 0 and idx_goodpoints[j] != 0:
				points[i,-1] = new_points[j]
				j+=1

			elif points[i,-2:-1] != 0 and idx_goodpoints[j] == 0:
				j+=1

		return points

	def add_head(self, gray, rect):

		tracker1 = dlib.correlation_tracker()
		tracker2 = dlib.correlation_tracker()

		(x, y, w, h) = rect_to_boundingbox(rect)
		x = np.ceil(x + 0.20 * w).astype(int)
		w = np.ceil(0.5 * w).astype(int)
		y = np.ceil(y - .20 * h).astype(int)
		h = np.ceil(0.85 * h).astype(int)

		# Rect1 = Forehead
		h1 = np.ceil(0.25 * h).astype(int)

		# Rect2 = Mouth and chin
		y2 = np.ceil(y + .80 * h).astype(int)
		h2 = np.ceil(0.5 * h).astype(int)

		size1 = (w, h1)
		size2 = (w, h2)

		gray1 = gray[y:y + h1, x:x + w]
		gray2 = gray[y2:y2 + h2, x:x + w]

		p01 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
		p02 = cv2.goodFeaturesToTrack(gray2, mask=None, **feature_params)

		try:
			tracker1.start_track(gray, dlib.rectangle(x, y, x + w, y + h1))
			tracker2.start_track(gray, dlib.rectangle(x, y2, x + w, y2 + h2))

			self.heads.append({
				'y1': p01[:, 0, 1],
				'y2': p02[:, 0, 1],
				'p01': p01,
				'p02': p02,
				'size1': size1,
				'size2': size2,
				'gray1': gray1,
				'gray2': gray2,
				'tracker1': tracker1,
				'tracker2': tracker2,
			})

		except(TypeError):
			pass

	def update_head(self, head, gray, mask, frame):

		trackingQuality1 = head['tracker1'].update(gray)
		trackingQuality2 = head['tracker2'].update(gray)

		bbox1 = rect_to_boundingbox(head['tracker1'].get_position())
		x1, y1, w1, h1 = [int(i) for i in bbox1]
		w1, h1 = head['size1']

		bbox2 = rect_to_boundingbox(head['tracker2'].get_position())
		x2, y2, w2, h2 = [int(i) for i in bbox2]
		w2, h2 = head['size2']

		new_gray1 = gray[y1:y1 + h1, x1:x1 + w1]
		new_gray2 = gray[y2:y2 + h2, x2:x2 + w2]
		cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0))
		cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0))

		if trackingQuality1 >= 8 or trackingQuality2 >= 8:

			try:
				p11, st1, err1 = cv2.calcOpticalFlowPyrLK(head['gray1'], new_gray1, head['p01'], None, **lk_params)
				p12, st2, err2 = cv2.calcOpticalFlowPyrLK(head['gray2'], new_gray2, head['p02'], None, **lk_params)

				# Select good points
				good_new1 = p11[st1 == 1]
				good_old1 = head['p01'][st1 == 1]
				good_new2 = p12[st2 == 1]
				good_old2 = head['p02'][st2 == 1]

			except(TypeError, cv2.error):
				return head, mask, frame, False

			# draw the tracks
			for i, (new, old) in enumerate(zip(good_new1, good_old1)):
				a, b = new.ravel()
				c, d = old.ravel()
				mask[y1:y1 + h1, x1:x1 + w1] = cv2.line(mask[y1:y1 + h1, x1:x1 + w1], (a, b), (c, d), color[i].tolist(), 2)
				frame[y1:y1 + h1, x1:x1 + w1] = cv2.circle(frame[y1:y1 + h1, x1:x1 + w1], (a, b), 5, color[i].tolist(), -1)

			# draw the tracks
			for i, (new, old) in enumerate(zip(good_new2, good_old2)):
				a, b = new.ravel()
				c, d = old.ravel()
				mask[y2:y2 + h2, x2:x2 + w2] = cv2.line(mask[y2:y2 + h2, x2:x2 + w2], (a, b), (c, d), color[i].tolist(), 2)
				frame[y2:y2 + h2, x2:x2 + w2] = cv2.circle(frame[y2:y2 + h2, x2:x2 + w2], (a, b), 5, color[i].tolist(), -1)

			head['gray1'] = new_gray1.copy()
			head['gray2'] = new_gray2.copy()
			head['p01'] = good_new1.reshape(-1, 1, 2)
			head['p02'] = good_new2.reshape(-1, 1, 2)
			head['y1'] = self.concat_points(head['y1'], p11[:, 0, 1], st1)
			head['y2'] = self.concat_points(head['y2'], p12[:, 0, 1], st2)

			return head, mask, frame, True

		else:
			return head, mask, frame, False


	def process_tracker(self):

		cap = cv2.VideoCapture(0)

		while 1:

			_, frame = cap.read()
			frame = cv2.flip(frame, 1)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			rects = self.detector(gray, 0)

			if not self.tracking:
				print('\nRestart tracking...')
				self.y = []
				self.tic = time.time()
				mask = np.zeros_like(frame)
				self.heads = []
				for i, rect in enumerate(rects):
					self.add_head(gray, rect)
				self.tracking = True

			elif len(rects) > len(self.heads):
				n = len(self.heads) - len(rects)
				for i, rect in enumerate(rects[n:]):
					self.add_head(gray, rect)

			for head in self.heads:
				head, mask, frame, updated = self.update_head(head, gray, mask, frame)
				if not updated:
					self.heads.remove(head)

			if len(self.heads) == 0:
				self.tracking = False

			frame = cv2.add(frame, mask)
			cv2.imshow('frame', frame)

			self.toc = time.time()
			print("Elapsed time: {:.2f} sec".format(self.toc - self.tic))
			self.fps.update()

			for head in self.heads:
				self.y.append(np.concatenate((head['y1'], head['y2']), axis=0))
				self.signal.main_loop(self.y, self.tic, self.toc)

			if cv2.waitKey(5) & 0xFF == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()

		self.fps.stop()
		print("\n[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))


if __name__ == '__main__':

	# ==================================================================================================================
	parser = argparse.ArgumentParser(description='Signal Processing of facial tracked points (head_tracker Subscriber Node)')
	parser.add_argument('--process_time', type=int, default=20, help='Time of acquisition')
	parser.add_argument('--lowcut', type=int, default=0.75, help='Lowcut frequency for Butterworth filter')
	parser.add_argument('--highcut', type=int, default=5, help='Highcut frequency for Butterworth filter')
	parser.add_argument('--Fs', type=int, default=250, help='Interpolated frame rate (Hz)')
	parser.add_argument('--alpha', type=int, default=0.25, help='% of points to discard')

	args = parser.parse_args()
	# ==================================================================================================================

	tracked_points = TRACKER(SIGNAL(args))
	tracked_points.process_tracker()
