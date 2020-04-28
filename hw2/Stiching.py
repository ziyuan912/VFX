import numpy as np
import cv2
import argparse
from scipy.ndimage import filters
import os
from tqdm import tqdm_notebook as tqdm

def read_img(filename):
	imgs = list()
	f = list()
	file = open(filename)
	lines = file.readlines()
	file.close()
	for i in range(len(lines)):
		if "#file " in lines[i]:
			imgs.append(cv2.imread(lines[i].split()[-1]))
		if "#f " in lines[i]:
			f.append(float(lines[i].split()[-1]))
	return imgs, f

def warping_imgs(imgs, fs):
	for i in len(imgs):
		warping(imgs[i], fs[i])

def warping(img, f):
	output = np.zeros(img.shape, dtype=int)
	y0 = img.shape[0] // 2
	x0 = img.shape[1] // 2
	for x in range(img.shape[1]):
		for y in range(img.shape[0]):
			x2 = f * np.tan((x - x0) / f)
			y2 = (y - y0)*np.sqrt(x2*x2 + f*f) / f
			x2 = int(x2 + x0)
			y2 = int(y2 + y0)
			if x2 > 0 and x2 < img.shape[1] and y2 > 0 and y2 < img.shape[0]:
				output[y][x] = img[y2][x2]
			else:
				output[y][x] = (0, 0, 0)
	cv2.imwrite("hi.png", output)


def main():
	parser = argparse.ArgumentParser(description='Process some images to do image stitching.')
	parser.add_argument("--file", help="input image feature file name")
	parser.add_argument("--output", help="the output file file name", default="./output/output.jpg")
	args = parser.parse_args()

	imgs, fs = read_img(args.file)
	warping(imgs[0], fs[0])
	#imgs = warping(imgs, fs)
if __name__ == '__main__':
	main()