import cv2
import json
import argparse
import numpy as np
import os, glob
from collections import defaultdict


def concat_frames(frames):
	if len(frames)==1:
		frame = frames[0]
	elif len(frames)==2:
		frame = np.hstack(frames)
	elif len(frames) == 3:
		frame1 = np.hstack(frames[:2])
		frame2 = np.hstack([frames[2], frames[2].copy()*0])
		frame = np.vstack([frame1, frame2])
	elif len(frames) == 4:
		frame1 = np.hstack(frames[:2])
		frame2 = np.hstack(frames[2:])
		frame = np.vstack([frame1, frame2])
	elif len(frames) == 5:
		frame1 = np.hstack(frames[:3])
		frame2 = np.hstack([frames[3], frames[4], frames[3].copy()*0])
		frame = np.vstack([frame1, frame2])
	elif len(frames) == 6:
		frame1 = np.hstack(frames[:3])
		frame2 = np.hstack(frames[3:])
		frame = np.vstack([frame1, frame2])

	return frame


def main(args):

	areas = {}
	with open(f"Dataset{args.dataset}/area.txt") as f:
		lines = f.read().split('\n')
		aids = eval(lines[0])
		for cid in range(1, len(aids)+1):
			areas[cid] = {}
			for aid in aids[cid-1]:
				areas[cid][aid] = eval(lines[aid])

	vcaps = []
	for key in areas.keys():
		p = f'../../../nlpr/data/video/dataset{args.dataset}/Cam{key}.avi'
		vcaps.append( ( cv2.VideoCapture(p), areas[key] ) )


	while True:
		frames = []
		for (vcap, areas) in vcaps:
			ret, frame = vcap.read()
			if not ret:
				break

			mask = np.zeros((frame.shape), dtype=np.uint8)
			for key in areas.keys():
				area = np.array(areas[key])
				mask = cv2.fillPoly(mask, [area], (0, 0,255))
				cv2.polylines(img=frame, pts=[area], isClosed=True, color=(0,0,255), thickness=2)
				frame = cv2.addWeighted(frame, 1, mask, 0.1, 0)
				#cv2.circle(frame, (area[4][0], area[4][1]), radius=5, color=(0, 255, 0), thickness=-1)
			frames.append(frame)

		nframe = concat_frames(frames)
		cv2.imshow('cam', nframe)

		if cv2.waitKey(1) == ord('q'):
			break


	cv2.destroyAllWindows()






if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default=1, type=int)
	args = parser.parse_args()

	main(args)
	





