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
	warp_img = list()
	for i in range(len(imgs)):
		warp_img.append(warping(imgs[i], fs[i]))
	return warp_img

def warping(img, f):
	output = np.zeros(img.shape, dtype=int)
	y0 = img.shape[0] // 2
	x0 = img.shape[1] // 2
	leftmost, rightmost, upmost, downmost = (img.shape[1] - 1, 0, img.shape[0] - 1, 0)
	for x in range(img.shape[1]):
		for y in range(img.shape[0]):
			x2 = f * np.tan((x - x0) / f)
			y2 = (y - y0)*np.sqrt(x2*x2 + f*f) / f
			x2 = int(x2 + x0)
			y2 = int(y2 + y0)
			if x2 > 0 and x2 < img.shape[1] and y2 > 0 and y2 < img.shape[0]:
				output[y][x] = img[y2][x2]
				if x < leftmost:
					leftmost = x
				if x > rightmost:
					rightmost = x
				if y < upmost:
					upmost = y
				if y > downmost:
					downmost = y
			else:
				output[y][x] = (0, 0, 0)
	return output[:, leftmost:rightmost]

def blending(img1, img2, w):
	output = np.zeros((img1.shape[0], img1.shape[1] + img2.shape[1] - w, 3), dtype = int)
	output[:, :img1.shape[1] - w] = img1[:, :img1.shape[1] - w]
	output[:, img1.shape[1]:] = img2[:, img2.shape[1] - w]
	for i in range(w):
		output[:, img1.shape[1] - w + i] = int(img1[:, img1.shape[1] - w + i] * (1 - float(1/w)) + img2[:, i] * (float(1/w)))
	cv2.imwrite("blending.png", output)
	return output

def main():
	parser = argparse.ArgumentParser(description='Process some images to do image stitching.')
	parser.add_argument("--file", help="input image feature file name")
	parser.add_argument("--output", help="the output file file name", default="./output/output.jpg")
	args = parser.parse_args()

	imgs, fs = read_img(args.file)
	#warping(imgs[0], fs[0])
	imgs = warping_imgs(imgs, fs)
if __name__ == '__main__':
	main()