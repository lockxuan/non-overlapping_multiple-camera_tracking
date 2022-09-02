import cv2



def df2bbox(bbox_df, frame, w=0, h=0):
	xyxy = []
	persons = []
	for person_id in list(bbox_df["pid"]):
		bbox_one_person_frame = bbox_df[bbox_df["pid"] == person_id]
		tlwh = bbox_one_person_frame.values.tolist()[0][3:7]
		
		if tlwh[2] > w and tlwh[3] > h:
			tlbr = list(map(int, [tlwh[0], tlwh[1], tlwh[0]+tlwh[2], tlwh[1]+tlwh[3]]))
			extract_frame = frame[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]]
			xyxy.append(tlbr)
			persons.append(extract_frame)

	return persons, xyxy



def draw_detection_bbox(frame, detections):
	for det in detections:
		color = (255, 200, 200)
		bbox = list(map(int, det[:4]))

		cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)


def draw_track_bbox(frame, bboxes):
	for bbox in bboxes:
		color = (100, 100, 255)

		cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
		cv2.putText(frame, str(bbox[4]), (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA)


def xyxy2tlwh(xyxy):
	tlwh = xyxy[:4]
	tlwh[2] = tlwh[2] - tlwh[0]
	tlwh[3] = tlwh[3] - tlwh[1]

	return tlwh




