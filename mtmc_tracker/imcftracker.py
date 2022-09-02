import cv2
import csv
import math
import pickle
import os, glob
import json
import torch
import pandas as pd
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict
from torchreid.utils import FeatureExtractor

from .tracklets import Tracklet
from .mct_utils import createFeatures, compute_delta, trklets_sim, motion_sim, get_area




class IMCFTracker:
	def __init__(self, did):
		self.graph = nx.DiGraph() # Directional graph
		self.trklets_ent = []	  # Tracklets (enter node)
		self.trklets_ext = []	  # Tracklets (exit node)

		# dataset parameters
		self.did = did 			  # dataset id
		self.topo = None		  # topology
		self.areas = None
		self.max_time = 99999	  # wait time (max)
		self.min_time = 0 		  # wait time (min)

		# single camera pid -> multiple camera pid
		self.inplace = defaultdict(list)
		self.replace = defaultdict(list)


	def loadParam(self, param_path):
		areas = {}
		with open(os.path.join(param_path, f'Dataset{self.did}', 'area.txt')) as f:
			lines = f.read().split('\n')
			aids = eval(lines[0])
			for cid in range(1, len(aids)+1):
				areas[cid] = {}
				for aid in aids[cid-1]:
					aid = int(aid)
					areas[cid][aid] = eval(lines[aid])
		
		topo = []
		with open(os.path.join(param_path, f'Dataset{self.did}', 'topology.txt')) as f:
			lines = f.read().split('\n')
			for l in lines:
				topo.append([ int(c) for c in l.split(' ') ])

		with open(os.path.join(param_path, f'Dataset{self.did}', 'param.json')) as f:
			data = json.load(f)
			self.max_time = data['max_time']
			self.min_time = data['min_time']

		self.areas = areas
		self.topo = topo


	def loadFeatures(self, vid_path, sct_path, folder='features'):
		print('[INFO] loading features...', end='\r')

		if not os.path.isdir(folder): # not a good checking method
			createFeatures(self.did, vid_path, sct_path, folder)

		csvpaths = glob.glob(os.path.join(sct_path, f'Dataset{self.did}', '*.csv'))
		for csvpath in csvpaths:
			cid = int(csvpath.split('/')[-1][3])
			df = pd.read_csv(csvpath)
			pids = list(set(df.pid.tolist()))

			for pid in pids:
				person_df = df[df.pid == pid]
				pickle_path = os.path.join(folder, f'Dataset{self.did}', f'Cam{cid}', f'{pid}.pkl')
				with open(pickle_path, "rb") as pickle_file:
					features = pickle.load(pickle_file)

				x = person_df.values.tolist()[0][3:7]
				x[1] += x[0]
				x[3] += x[2]
				t = person_df.fid.tolist()

				area = get_area(person_df, t[0], t[-1], self.areas[cid])
				enter_node = Tracklet(pos=x, cid=cid, time=t, features=features, pid=pid, enter=True, area=area[0])
				exit_node = Tracklet(pos=x, cid=cid, time=t, features=features, pid=pid, enter=False, area=area[1])
				self.trklets_ent.append(enter_node)
				self.trklets_ext.append(exit_node)

				self.graph.add_edge(enter_node, exit_node, weight=-100, capacity=1)
				self.graph.add_edge('S', enter_node, weight=0, capacity=1)
				self.graph.add_edge(exit_node, 'T', weight=0, capacity=1)

		print('[INFO] loading features... Done')


	def process(self):
		self.trklets_ent.sort(key=lambda trklet: trklet.time[0])
		self.trklets_ext.sort(key=lambda trklet: trklet.time[-1])

		self.delta_mean, self.delta_var = compute_delta(self.trklets_ent, self.trklets_ext, wait_time=self.max_time, topo=self.topo)
		print(f'\n[INFO] ∆μ (mean): {self.delta_mean}, ∆σ (var): {self.delta_var}\n')

		self.create_edge()

		iter = 0
		v = 1
		print('[INFO] iterating min cost flow...')
		while True:
			print(f'[RUNNING] iter: {iter+1}', end='\r')
			self.graph.add_node("S", demand=-v)
			self.graph.add_node("T", demand=v)

			flowDict = nx.min_cost_flow(self.graph)
			minFlowCost = nx.min_cost_flow_cost(self.graph)
			self.update(flowDict, iter)
			iter += 1
			if len(self.graph.nodes) == 2:
				break
		print('[INFO] iterating min cost flow... Done')


	def update(self, flowDict, mpid):
		del_nodes = set()
		for i in flowDict.keys():
			if i == 'S' or i =='T':
				continue
			for j in flowDict[i].keys():
				if j == 'S' or j =='T':
					continue

				if flowDict[i][j] > 0:
					del_nodes.add(i)
					del_nodes.add(j)

		for node in list(del_nodes):
			self.inplace[node.cid].append(node.pid)
			self.replace[node.cid].append(mpid)
			self.graph.remove_node(node)


	def output_result(self, dataset, sct_path, output_path):
		print(f'[INFO] output result of datset{dataset} to {output_path}')
		for cid in self.inplace.keys():
			inp = self.inplace[cid]
			rep = self.replace[cid]

			cam_file = os.path.join(sct_path, f'Dataset{dataset}', f'Cam{cid}.csv')
			cam_df = pd.read_csv(cam_file)
			cam_df.pid.replace(inp, rep, inplace=True)

			output_file = os.path.join(output_path, f'Dataset{dataset}', f'Cam{cid}.csv')
			os.makedirs(os.path.dirname(output_file), exist_ok=True)
			cam_df.to_csv(output_file, index=False)



	def create_edge(self, thres=0.5, k1=1.0, k2=0):
		print('[INFO] creating weights between all nodes...', end='\r')
		for i, trklet_i in enumerate(self.trklets_ext):
			for j, trklet_j in enumerate(self.trklets_ent):
				inv_t = trklet_j.time[0] - trklet_i.time[-1]
				c1 = trklet_i.cid
				c2 = trklet_j.cid
				ext = trklet_i.area
				ent = trklet_j.area

				if self.min_time < inv_t < self.max_time and ext != None and ent != None and self.topo[ext-1][ent-1]==1:
					mean = 0
					var = 1
					if c1 == c2:
						mean = self.delta_mean
						var = self.delta_var

					non_a_sim = trklets_sim(trklet_i.features, trklet_j.features)
					a_sim = (non_a_sim - mean) * var
					if a_sim < thres:
						continue

					m_sim = motion_sim(trklet_i, trklet_j)

					final_sim = -k1 * math.log(a_sim, 10) - k2 * math.log(m_sim, 10)
					final_sim = int(100 * final_sim)
					self.graph.add_edge(trklet_i, trklet_j,  weight=final_sim, capacity=1)

		print('[INFO] creating weights between all nodes... Done')















