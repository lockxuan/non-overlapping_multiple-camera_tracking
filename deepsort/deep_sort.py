import numpy as np
import torch

from .tracker import Tracker
from .matching import NearestNeighborDistanceMetric


class Detection(object):
	def __init__(self, bbox, feature):
		self.tlwh = np.asarray(bbox).astype(float)
		self.feature = np.asarray(feature)

	def to_tlbr(self):
		ret = self.tlwh.copy()
		ret[2:] += ret[:2]
		return ret

	def to_xyah(self):
		""" to (center x, center y, aspect ratio, height) """
		ret = self.tlwh.copy()

		ret[:2] += ret[2:] / 2
		ret[2] /= ret[3]
		return ret


		


class DeepSort(object):
	def __init__(self, width, height, max_dist=0.8):
		
		max_cosine_distance = max_dist

		metric = NearestNeighborDistanceMetric('cosine')
		self.tracker = Tracker(metric=metric)
		self.width, self.height = width, height

		print('load deep sort ok')



	def update(self, detections, features):
		bboxes = [list(map(int, det[:4])) for det in detections]
		bbox_tlwh = self.tlbr_to_tlwh(bboxes)
		detections = [ Detection(bbox_tlwh[i], features[i]) for i in range(len(bboxes)) ]

		# update tracker
		self.tracker.predict()
		self.tracker.update(detections)

		# output bbox identities
		outputs = []
		for track in self.tracker.tracks:
			if not track.is_confirmed() or track.time_since_update > 1:
				continue

			box = track.to_tlwh()
			x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
			track_id = track.track_id
			outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))

		if len(outputs) > 0:
			outputs = np.stack(outputs,axis=0)
		return outputs




	def tlbr_to_tlwh(self, tlbr):
		if len(tlbr) == 0:
			return []
		tlbr = np.asarray(tlbr)
		tlwh = tlbr.copy()

		tlwh[:, 2] = tlbr[:, 2] - tlbr[:, 0]
		tlwh[:, 3] = tlbr[:, 3] - tlbr[:, 1]

		return tlwh

	def _tlwh_to_xyxy(self, tlwh):
		x,y,w,h = tlwh
		x1 = max(int(x),0)
		x2 = min(int(x+w),self.width-1)
		y1 = max(int(y),0)
		y2 = min(int(y+h),self.height-1)
		return x1,y1,x2,y2




		


