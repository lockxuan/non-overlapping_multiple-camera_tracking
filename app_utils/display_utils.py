import cv2
import threading
import random
import numpy as np
import time




class video_capture:
	def __init__(self, video_path, total_cams):
		self.video_path = video_path
		self.video_cap = None

		### need to fix, not using sleep way? ###
		self.sleep_time = max(total_cams * 0.015, 1/30)

		self.frame = []
		self.status = True
		self.stopped = False
		self.frame_no = 0

	def start(self):
		self.video_cap = cv2.VideoCapture(self.video_path)
		self.process = threading.Thread(target=self.get_frame, daemon=True, args=()).start()
		
	def get_frame(self):
		while True:
			if not self.status:
				self.stop()
			else:
				if not self.stopped:
					self.status, self.frame = self.video_cap.read()
					self.frame_no += 1
					time.sleep(self.sleep_time)

	def stop(self):
		self.stopped = True
		self.video_cap.release()
		print('----> release video cap correctly')

	def pause(self):
		self.stopped = not self.stopped

	def set_frame(self, x):
		self.frame_no = x
		self.video_cap.set(1, x)
		self.status, self.frame = self.video_cap.read()





def create_color(df):
	pid_color = {}

	pids = df['pid'].astype('category').unique()
	for pid in pids:
		color = [random.randint(0, 255) for i in range(3)]
		pid_color[pid] = (color[0], color[1], color[2])
	return pid_color



def DrawBbox(frame, bbox_frame, person_color):
	for person_id in list(bbox_frame.pid):
		color = person_color[person_id]

		bbox_one_person_frame = bbox_frame[bbox_frame["pid"] == person_id]
		for p in bbox_one_person_frame.values.tolist():
			tlwh = p[3:7]

			cv2.rectangle(frame, (tlwh[0], tlwh[1]), (tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]), color=color, thickness=2)
			cv2.putText(frame, str(person_id), (tlwh[0], tlwh[1]-5), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA)

#### to fix ###
def concat_frames(frames):
	l = len(frames)
	if l == 1:
		frame = frames[0]
	elif l == 2:
		frame = np.hstack(frames)
	elif l == 3:
		frame1 = np.hstack(frames[:2])
		frame2 = np.hstack([frames[2], frames[2].copy()*0])
		frame = np.vstack([frame1, frame2])
	elif l == 4:
		frame1 = np.hstack(frames[:2])
		frame2 = np.hstack(frames[2:])
		frame = np.vstack([frame1, frame2])
	elif l == 5:
		frame1 = np.hstack(frames[:3])
		frame2 = np.hstack([frames[3], frames[4], frames[3].copy()*0])
		frame = np.vstack([frame1, frame2])
	elif l == 6:
		frame1 = np.hstack(frames[:3])
		frame2 = np.hstack(frames[3:])
		frame = np.vstack([frame1, frame2])

	return frame





