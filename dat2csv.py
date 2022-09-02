import argparse
import os
import glob
import csv





def argument():
	parser = argparse.ArgumentParser()

	parser.add_argument('--source', default='../nlpr/data/annotation', type=str)
	parser.add_argument('--dataset', default='1', type=str)

	args = parser.parse_args()
	return args



def main(args):

	dats = glob.glob(os.path.join(args.source, f'Dataset{args.dataset}', '*.dat'))

	for dat in dats:
		output_file = open(dat.replace('.dat', '.csv'), 'w', newline='')
		writer = csv.writer(output_file)
		writer.writerow(["cid", "fid", "pid", "x1", "y1", "w", "h"])
		with open(dat) as f:
			lines = f.read().split('\n')
			for line in lines:
				if line == '':
					continue
				row = [ int(c) for c in line.split('  ') ]
				writer.writerow(row)




if __name__ == '__main__':
	args = argument()
	main(args)




