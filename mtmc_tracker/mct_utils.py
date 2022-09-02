import torch
import os, glob
import cv2
import pickle
import pandas as pd
import numpy as np

from functools import lru_cache
from tqdm import tqdm
from torchreid.utils import FeatureExtractor





def run_extract(reid_model, frames):
	features = reid_model(frames)
	centroid = np.array(features).mean(axis=0)
	return centroid


@lru_cache(16000)
def getFrame(v_cap, fid):
	v_cap.set(1, fid)
	frame = v_cap.read()[1]
	return frame

def createFeatures(did, vid_path, sct_path, output_folder, reid_path='./osnet_ain_x1_0_cosine.pth', sample_num=25, Di=5):
	print('[INFO] start creating features ...', end='\r')

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	reid_model = FeatureExtractor(model_name='osnet_ain_x1_0', model_path=reid_path, device=device)

	csvpaths = glob.glob(os.path.join(sct_path, f'Dataset{did}', '*.csv'))
	for csvpath in csvpaths:
		cid = int(csvpath.split('/')[-1][3])
		cam_df = pd.read_csv(csvpath)

		video_path = os.path.join(vid_path, f'Dataset{did}', f'Cam{cid}.avi')
		vcap = cv2.VideoCapture(video_path)

		for pid in tqdm(list(set(cam_df.pid.tolist()))):
			person_df = cam_df[cam_df.pid == pid]
			_frames_no = person_df.fid.tolist()
			frames_idx = np.linspace(0, len(_frames_no)-1, num=sample_num, dtype=int) \
							if len(_frames_no) > sample_num else range(len(_frames_no))
			frames_no = [_frames_no[idx] for idx in frames_idx]

			if len(frames_no) >= Di:
				fid_linspace = np.array_split(np.array(frames_no), Di)
			else:
				fid_linspace = [ [f] for f in frames_no]

			features = []
			for fid_lin in fid_linspace:
				frames = []
				for fid in fid_lin:
					frame = getFrame(vcap, fid)
					bbox = person_df[person_df.fid == fid].values.tolist()[0][3:7]
					extract_frame = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
					frames.append(extract_frame)

				features.append( run_extract(reid_model, frames) )

			pickle_path = os.path.join(output_folder, f'Dataset{did}', f'Cam{cid}', f'{pid}.pkl')
			os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
			with open(pickle_path, "wb") as pickle_file:
				pickle.dump(features, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

	print('[INFO] start creating features ... Done')



def get_area(person_df, start, end, areas):
	enter_df = person_df[person_df.fid == start].values.tolist()[0][3:]
	exit_df = person_df[person_df.fid == end].values.tolist()[0][3:]
	enter = [enter_df[0]+enter_df[2]//2, enter_df[1]+enter_df[3]//2]
	exit = [exit_df[0]+exit_df[2]//2, exit_df[1]+exit_df[3]//2]
	enter_id = None
	exit_id = None

	for aid in areas.keys():
		area = areas[aid]
		if area[0][0] < enter[0] < area[2][0] and \
			area[0][1] < enter[1] < area[2][1]:
				enter_id = aid
		if area[0][0] < exit[0] < area[2][0] and \
			area[0][1] < exit[1] < area[2][1]:
				exit_id = aid
	return enter_id, exit_id



def l2_distance(a, b):
	a = np.asarray(a) / np.linalg.norm(a, axis=0, keepdims=True)
	b = np.asarray(b) / np.linalg.norm(b, axis=0, keepdims=True)

	dist = np.linalg.norm(a-b)
	print(dist)
	return dist

def cosine_distance(a, b, normalize=True):
	if normalize:
		a = np.asarray(a) / np.linalg.norm(a, axis=0, keepdims=True)
		b = np.asarray(b) / np.linalg.norm(b, axis=0, keepdims=True)

	return np.dot(a, b.T).min(axis=0)



def compute_mug(features_i, features_j, method='cosine'):
	if method == 'cosine':
		distance = cosine_distance
	else:
		distance = l2_distance
	maxSim = 0.0
	minSim = 1.0
	for fi in features_i:
		for fj in features_j:
			maxSim = max(maxSim, distance(fi, fj))
			minSim = min(minSim, distance(fi, fj))

	return maxSim - minSim

def trklets_sim(features_i, features_j, method='cosine'):
	if method == 'cosine':
		distance = cosine_distance
	else:
		distance = l2_distance

	dist = 0
	for fi in features_i:
		for fj in features_j:
			dist += distance(fi, fj)
	dist /= (len(features_i) * len(features_j))
	
	return dist


def compute_delta(trklets_ent, trklets_ext, wait_time, topo=None, epsilon=0.4):
	print('[INFO] computing ∆μ (mean) and ∆σ (var)...', end='\r')
	sum_sim_s = 0.0
	sum_sim_c = 0.0
	sim_pair_s = 0
	sim_pair_c = 0

	for trklet_i in trklets_ext:
		for trklet_j in trklets_ent:
			inv_t = trklet_j.time[0] - trklet_i.time[-1]
			c1 = trklet_i.cid
			c2 = trklet_j.cid
			ext = trklet_i.area
			ent = trklet_j.area
			if inv_t <= 0 or inv_t >= wait_time:
				continue

			mug = compute_mug(trklet_i.features, trklet_j.features)
			if mug < epsilon and ext != None and ent != None and topo[ext-1][ent-1] == 1:
				tmp_sim = trklets_sim(trklet_i.features, trklet_j.features)
				if c1 == c2:
					sum_sim_s += tmp_sim
					sim_pair_s += 1
				else:
					sum_sim_c += tmp_sim
					sim_pair_c += 1

	sim_s = sum_sim_s / sim_pair_s
	sim_c = sum_sim_c / sim_pair_c

	sum_var_s = 0.0
	sum_var_c = 0.0

	for trklet_i in trklets_ext:
		for trklet_j in trklets_ent:
			inv_t = trklet_j.time[0] - trklet_i.time[-1]
			c1 = trklet_i.cid
			c2 = trklet_j.cid
			ext = trklet_i.area
			ent = trklet_j.area
			if inv_t <= 0 or inv_t >= wait_time:
				continue

			mug = compute_mug(trklet_i.features, trklet_j.features)
			if mug < epsilon and ext != None and ent != None and topo[ext-1][ent-1] == 1:
				tmp_sim = trklets_sim(trklet_i.features, trklet_j.features)
				if c1 == c2:
					sum_var_s += (tmp_sim - sim_s) ** 2
				else:
					sum_var_c += (tmp_sim - sim_c) ** 2

	var_s = sum_var_s / sim_pair_s
	var_c = sum_var_c / sim_pair_c

	delta_mean = sim_s - sim_c
	delta_var = var_s / var_c

	print('[INFO] computing ∆μ (mean) and ∆σ (var)... Done')
	return delta_mean, delta_var


def motion_sim(trklet_i, trklet_j):
	return 1






