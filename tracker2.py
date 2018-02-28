import numpy as np
import cv2
import dlib
from tools import rect_to_boundingbox
import pickle


def concat_points(points, new_points, idx_goodpoints):

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


detector = dlib.get_frontal_face_detector()
tracker1 = dlib.correlation_tracker()
tracker2 = dlib.correlation_tracker()

cap = cv2.VideoCapture(0)

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
tracking = False

log = []
k = 0

trackingQuality1 = 0
trackingQuality2 = 0

while 1:

	_, frame = cap.read()
	frame = cv2.flip(frame,1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	if not tracking:

		rects = detector(gray, 0)

		if rects:
			mask = np.zeros_like(frame)

			(x, y, w, h) = rect_to_boundingbox(rects[0])
			x = np.ceil(x + 0.20 * w).astype(int)
			w = np.ceil(0.5 * w).astype(int)
			y = np.ceil(y - .20 * h).astype(int)
			h = np.ceil(0.85 * h).astype(int)

			# Rect1 = Forehead
			h1 = np.ceil(0.25 * h).astype(int)

			# Rect2 = Mouth and chin
			y2 = np.ceil(y + .80 * h).astype(int)
			h2 = np.ceil(0.5 * h).astype(int)

			size1 = (w,h1)
			size2 = (w, h2)

			gray1 = gray[y:y + h1, x:x + w]
			gray2 = gray[y2:y2 + h2, x:x + w]

			p01 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
			p02 = cv2.goodFeaturesToTrack(gray2, mask=None, **feature_params)

			try:
				log.append({
					'head_id': 0,
					'tracking_id': k,
					'p01x': p01[:,0,0],
					'p01y': p01[:,0,1],
					'p02x': p02[:,0,0],
					'p02y': p02[:,0,1],
				})
			except(TypeError):
				tracking = False
				continue

			tracker1.start_track(gray, dlib.rectangle(x, y, x + w, y + h1))
			tracker2.start_track(gray, dlib.rectangle(x, y2, x + w, y2 + h2))

			tracking = True

	if tracking:
		trackingQuality1 = tracker1.update(gray)
		trackingQuality2 = tracker2.update(gray)

		bbox1 = rect_to_boundingbox(tracker1.get_position())
		x1, y1, w1, h1 = [int(i) for i in bbox1]
		w1, h1 = size1

		bbox2 = rect_to_boundingbox(tracker2.get_position())
		x2, y2, w2, h2 = [int(i) for i in bbox2]
		w2, h2 = size2

		new_gray1 = gray[y1:y1 + h1, x1:x1 + w1]
		new_gray2 = gray[y2:y2 + h2, x2:x2 + w2]
		cv2.rectangle(frame, (x1,y1), (x1+w1,y1+h1), (0,255,0))
		cv2.rectangle(frame, (x2,y2), (x2+w2,y2+h2), (0,255,0))

		if trackingQuality1 >= 8 or trackingQuality2 >= 8:

			try:
				p11, st1, err1 = cv2.calcOpticalFlowPyrLK(gray1, new_gray1, p01, None, **lk_params)
				p12, st2, err2 = cv2.calcOpticalFlowPyrLK(gray2, new_gray2, p02, None, **lk_params)

				# Select good points
				good_new1 = p11[st1 == 1]
				good_old1 = p01[st1 == 1]
				good_new2 = p12[st2 == 1]
				good_old2 = p02[st2 == 1]

			except(TypeError, cv2.error):
				tracking = False
				k += 1
				continue

			# draw the tracks
			for i, (new, old) in enumerate(zip(good_new1, good_old1)):
				a, b = new.ravel()
				c, d = old.ravel()
				mask[y1:y1+h1, x1:x1+w1] = cv2.line(mask[y1:y1+h1, x1:x1+w1], (a, b), (c, d), color[i].tolist(), 2)
				frame[y1:y1+h1, x1:x1+w1] = cv2.circle(frame[y1:y1+h1, x1:x1+w1], (a, b), 5, color[i].tolist(), -1)

			# draw the tracks
			for i, (new, old) in enumerate(zip(good_new2, good_old2)):
				a, b = new.ravel()
				c, d = old.ravel()
				mask[y2:y2+h2, x2:x2+w2] = cv2.line(mask[y2:y2+h2, x2:x2+w2], (a, b), (c, d), color[i].tolist(), 2)
				frame[y2:y2+h2, x2:x2+w2] = cv2.circle(frame[y2:y2+h2, x2:x2+w2], (a, b), 5, color[i].tolist(), -1)

			gray1 = new_gray1.copy()
			gray2 = new_gray2.copy()
			p01 = good_new1.reshape(-1, 1, 2)
			p02 = good_new2.reshape(-1, 1, 2)

			log[k]['p01x'] = concat_points(log[k]['p01x'], p11[:,0,0], st1)
			log[k]['p01y'] = concat_points(log[k]['p01y'], p11[:,0,1], st1)
			log[k]['p02x'] = concat_points(log[k]['p02x'], p12[:,0,0], st2)
			log[k]['p02y'] = concat_points(log[k]['p02y'], p12[:,0,1], st2)

			frame = cv2.add(frame, mask)

		else:
			tracking = False
			k += 1

	cv2.imshow('frame', frame)
	if cv2.waitKey(5) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

pickle.dump(log, open('log.p', 'wb'))


