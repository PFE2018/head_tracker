import numpy as np
import cv2
import dlib
from tools import rect_to_boundingbox, shape_to_array
import time
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
from landmarks_processor import *
import argparse
from biosppy import ecg
from scipy.io import loadmat

# Create some random colors
color = np.random.randint(0, 255, (100, 3))


class TRACKER:

    def __init__(self, signal):

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.y = []
        self.points = []
        self.nb_frame = 0
        self.tic0 = time.time()
        self.tic = time.time()
        self.toc = time.time()
        self.frame = [0]
        self.signal = signal
        self.bpm = []
        self.spectrums = []
        self.freq = []
        self.time = [0]
        self.start = False

    def get_image(self, msg):

        msg.encoding = "bgr8"
        # Convert ROS image to opencv image.
        try:
            bridge = CvBridge()
            # Need to change encoding
            self.frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

    def process_tracker(self):

        if len(self.frame) > 1 and not self.start:
            self.start = True
            self.tic0 = time.time()

        if self.start:

            frame = self.frame[:, 500:, :]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = self.detector(gray, 0)
            mask = np.zeros_like(frame)

            if len(rects) == 1:
                rect = rects[0]

                landmarks = self.predictor(gray, rect)
                landmarks = shape_to_array(landmarks)

                (x, y, w, h) = rect_to_boundingbox(rect)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # draw the tracks
                for i, point in enumerate(landmarks):
                    new = point
                    a, b = new.ravel()

                    if self.nb_frame > 0:
                        old = np.transpose(self.points[-1][i])
                        c, d = old.ravel()
                        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                    frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

                self.points.append(landmarks)
                self.y.append(landmarks[:,0])

                if self.toc - self.tic >= self.signal.args.process_time:
                    bpm, spectrums, freq = self.signal.main_loop(self.y, self.tic, self.toc)
                    self.bpm.append(bpm)
                    self.time.append(self.toc - self.tic0)
                    self.spectrums.append(spectrums)
                    self.freq.append(freq)
                    self.tic = time.time()
                    print('\t\t\t\t\t\t\t\t\t\t' + str(np.ceil(self.bpm[-1]).astype(int)) + ' beats per min')  # beats per minute

            else:
                self.tic = time.time()
                print('\n')
                #print('\t\t\t\t\t\t\t\t\t\t' + str(np.ceil(self.bpm[-1]).astype(int)) + ' beats per min')  # beats per minute

            self.nb_frame += 1
            frame = cv2.add(frame, mask)
            #cv2.imshow('frame', frame)

            self.toc = time.time()
            print("Elapsed time: {:.2f} sec     {:.2f} sec".format(self.toc - self.tic0, self.toc - self.tic))

            if cv2.waitKey(5) & 0xFF == ord('q'):
                sys.exit()

        #cv2.destroyAllWindows()

    def talker(self):

        # pub = rospy.Publisher('chatter', String, queue_size=10)
        rospy.init_node('points_tracker', anonymous=True)
        rospy.Subscriber("/kinect2/hd/image_color_rect", Image, self.get_image)
        rate = rospy.Rate(30)  # 10hz
        self.toc2 = time.time()
        delta = self.toc2 - self.tic0

        while not rospy.is_shutdown() and delta < 300:
            delta = self.toc2 - self.tic0
            self.toc2 = time.time()
            self.process_tracker()
            rate.sleep()

if __name__ == '__main__':
    # ==================================================================================================================
    parser = argparse.ArgumentParser(description='Signal Processing of facial tracked points (head_tracker Subscriber Node)')
    parser.add_argument('--process_time', type=int, default=5, help='Time of acquisition')
    parser.add_argument('--lowcut', type=int, default=0.75, help='Lowcut frequency for Butterworth filter')
    parser.add_argument('--highcut', type=int, default=5, help='Highcut frequency for Butterworth filter')
    parser.add_argument('--Fs', type=int, default=250, help='Interpolated frame rate (Hz)')
    parser.add_argument('--alpha', type=int, default=0.25, help='% of points to discard')

    args = parser.parse_args()
    # ==================================================================================================================

    tracked_points = TRACKER(SIGNAL(args))
    try:
        tracked_points.talker()
    except rospy.ROSInterruptException:
        pass

    mat = loadmat('REF_CHAISE_COUTURIER_ELODIE_2018_04_05_19_32.mat')
    out = ecg.ecg(signal=mat['data'][0], sampling_rate=mat['samplerate'][0][0], show=False)[6]

    items = True
    for j in range(0, 68):
        bpm = []
        #spectrum = []
        for i in range(0, len(tracked_points.bpm)):
            bpm.append(tracked_points.bpm[i][j])
            #spectrum.append(tracked_points.spectrums[i][j])

        fig = plt.figure()
        time = tracked_points.time[:len(bpm)]
        #plt.subplot(2, 1, 1)
        plt.plot(time, bpm)
        plt.plot(out[:300])
        plt.xlabel('Time (sec)')
        plt.ylabel('BPM')
        fig.savefig('results/results_' + str(j) + '.png')
        #plt.subplot(2, 1, 2)
        #plt.plot(tracked_points.freq[j], spectrum)

