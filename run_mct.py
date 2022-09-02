from mtmc_tracker.imcftracker import IMCFTracker

import os, glob
import argparse


def argument():
	parser = argparse.ArgumentParser()

	parser.add_argument('--source', default='../nlpr/data/video', type=str)
	parser.add_argument('--dataset', default=1, type=int)
	parser.add_argument('--sct_path', default='./sct_result', type=str)
	parser.add_argument('--params', default='./nlprmct', type=str)
	parser.add_argument('--features', default='./features', type=str)
	parser.add_argument('--output_path', default='./mct_result', type=str)


	args = parser.parse_args()
	return args


def main(args):

	Tracker = IMCFTracker(did=args.dataset)

	Tracker.loadParam(args.params)
	Tracker.loadFeatures(vid_path=args.source, sct_path=args.sct_path, folder=args.features)
	Tracker.process()
	Tracker.output_result(dataset=args.dataset, sct_path=args.sct_path, output_path=args.output_path)






if __name__ == '__main__':
	args = argument()
	main(args)





