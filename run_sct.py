import torch
import cv2
import argparse
import os, glob
import grpc
import pandas as pd
import numpy as np
from scipy.spatial import distance
from torchreid.utils import FeatureExtractor

from deepsort.deep_sort import DeepSort
from app_utils.sct_utils import df2bbox, draw_detection_bbox, draw_track_bbox, xyxy2tlwh

import csv



def argument():
	parser = argparse.ArgumentParser()

	parser.add_argument('--source', default='../nlpr/data', type=str)
	parser.add_argument('--dataset', default='1', type=str)
	parser.add_argument('--cid', default='1', type=str)

	parser.add_argument('--display', action='store_true')

	parser.add_argument('--output_path', default='sct_result', type=str)
	parser.add_argument('--reid_path', default='osnet_ain_x1_0_cosine.pth', type=str)

	args = parser.parse_args()

	return args




def load_video(source, dataset, camid):
	video_path = os.path.join(source, 'video', f'Dataset{dataset}', f'Cam{camid}.avi')
	video_cap = cv2.VideoCapture(video_path)

	annotation_path = os.path.join(source, 'annotation', f'Dataset{dataset}', f'Cam{camid}.csv')
	annotation = pd.read_csv(annotation_path)

	return video_cap, annotation

def load_reid_model(device, reid_path):
	reid_model = FeatureExtractor(
			model_name='osnet_ain_x1_0',
			model_path=reid_path,
			device=device
		)

	return reid_model

def run_extract(model, bbox_df, frame):
	persons, bboxes = df2bbox(bbox_df, frame)
	if len(persons) == 0:
		return [], []

	imgs = [cv2.resize(p, (128, 256), interpolation=cv2.INTER_AREA) for p in persons]
	features = model(imgs)

	return bboxes, features




def main(args):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	reid_model = load_reid_model(device, args.reid_path)

	video_cap, annotation_df = load_video(args.source, args.dataset, args.cid)
	fps = video_cap.get(cv2.CAP_PROP_FPS)
	print('[INFO] video fps: {}'.format(fps))

	total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

	ret, frame = video_cap.read()
	h, w, c = frame.shape
	deepsort = DeepSort(width=w, height=h)

	output_file = os.path.join(f'{args.output_path}', f'Dataset{args.dataset}', f'Cam{args.cid}.csv')
	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	output_csv = open(output_file, 'w', newline='')
	writer = csv.writer(output_csv)
	writer.writerow(["cid", "fid", "pid", "x1", "y1", "w", "h"])

	current_frame = 0
	while ret:
		key = cv2.waitKey(1)
		if key & 0xFF == ord('q'):
			break
		elif key == ord(' '):
			continue

		# use groundtruth bounding box
		bbox_df = annotation_df[annotation_df['fid'] == current_frame]
		detections, features = run_extract(reid_model, bbox_df, frame)

		outputs = deepsort.update(detections, features)
		if len(detections) != 0:
			if args.display:
				draw_detection_bbox(frame, detections)
				draw_track_bbox(frame, outputs)

			for output in outputs:
				tlwh = xyxy2tlwh(output)
				writer.writerow([ args.cid, current_frame, output[4], tlwh[0], tlwh[1], tlwh[2], tlwh[3] ])

		text = 'cam: {}, {}/{}'.format(args.cid, current_frame, total_frames)
		if args.display:
			cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,100,100), 1, cv2.LINE_AA)
			cv2.imshow('cams', frame)

		ret, frame = video_cap.read()
		current_frame += 1

		print(text, end='\r')

	cv2.destroyAllWindows()




if __name__ == '__main__':
	args = argument()
	main(args)










