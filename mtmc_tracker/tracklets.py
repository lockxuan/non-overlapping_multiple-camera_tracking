



class Tracklet:
	def __init__(self, pos, cid, time, features, pid, enter, area):
		self.pos = pos				# position
		self.cid = cid				# camera id
		self.time = time			# time stamp
		self.features = features	# features
		self.pid = pid				# pif in sct
		self.enter = enter			# enter if True else exit (node)
		self.area = area			# the enter/exit area



