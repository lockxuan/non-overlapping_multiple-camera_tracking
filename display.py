import os
import argparse
import cv2
import pandas as pd
import numpy as np
import random

import threading
import time

from app_utils.display_utils import DrawBbox, concat_frames, video_capture, create_color


pid_color = {}


def ShowVideo(args):
	def change_frame(x):
		if video_caps[0].stopped == True:
			for cap in video_caps:
				cap.set_frame(x)
		else:
			for cap in video_caps:
				cap.pause()
				cap.set_frame(x)
			for cap in video_caps:
				cap.pause()


	video_caps = [None] * len(args.cams)
	total_frames = [None] * len(args.cams)
	annotaions = [None] * len(args.cams)

	### prepare video captures and annotation
	for i, cid in enumerate(args.cams):
		### load annotaion
		a_path = os.path.join(args.annotation, f'Dataset{args.dataset}', f'Cam{cid}.csv')
		annotaions[i] = pd.read_csv(a_path)

		pid_color = create_color(annotaions[i])
		if len(args.pid) > 0:
			annotaions[i] = annotaions[i][annotaions[i]['pid'].isin(args.pid)]

		### load video
		path = os.path.join(args.data_path, 'video', f'Dataset{args.dataset}', f'Cam{cid}.avi')
		video_caps[i] = video_capture(path, len(args.cams))
		video_caps[i].start()
		total_frames[i] = int(video_caps[i].video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

	frames = [None] * len(args.cams)
	Num = 0

	time.sleep(1)
	cv2.namedWindow('cams')
	cv2.createTrackbar('track bar','cams',0, total_frames[i], change_frame)
	cv2.setTrackbarPos('track bar','cams', 0)
	while True:
		key = cv2.waitKey(1)
		if key & 0xFF == ord('q'):
			for cap in video_caps:
				cap.stop()
			break
		elif key == ord(' '):
			for cap in video_caps:
				cap.pause()

		for i, (cap, cam_id) in enumerate(zip(video_caps, args.cams)):
			frames[i] = cap.frame
			if args.bbox:
				bbox_frame = annotaions[i][annotaions[i]['fid'] == cap.frame_no]
				DrawBbox(frames[i], bbox_frame, pid_color)

			text = 'cam: {}, {}/{}'.format(cam_id, cap.frame_no, total_frames[i])
			cv2.putText(frames[i], text, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,100,100), 1, cv2.LINE_AA)

		frame = concat_frames(frames)

		cv2.imshow('cams', frame)
		if video_caps[0].stopped == False:
			Num += 1
			cv2.setTrackbarPos('track bar','cams', cv2.getTrackbarPos('track bar','cams')+1)


	time.sleep(1)
	cv2.destroyAllWindows()



def main(args):
	print(f'[INFO] showing dataset: {args.dataset}, cameras: {args.cams}')
	ShowVideo(args)
	


def argument():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', default='../nlpr/data', type=str)
	parser.add_argument('--dataset', default='1', type=str)

	parser.add_argument('--annotation', type=str, default='../nlpr/data/annotation')
	parser.add_argument('--cams', type=str, default="1,2,3", help='choose cams to show')
	parser.add_argument('--pid', type=str, default='', help='person to focus')

	parser.add_argument('--bbox', action='store_true', help='show the annotations or not')

	args = parser.parse_args()
	args.cams = list(map(int,args.cams.split(",")))
	if args.pid != '':
		args.pid = list(map(int,args.pid.split(",")))
	if len(args.pid) > 0:
		args.bbox = True

	return args

if __name__ == '__main__':
	args = argument()
	main(args)








