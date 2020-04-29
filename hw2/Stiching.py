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

def DownSampling(Images, ScalePercent=10):
    """
    Resize the Images
    """
    width = int(Images[0].shape[1] * ScalePercent / 100)
    height = int(Images[0].shape[0] * ScalePercent / 100)
    dim = (width, height)
    ResizeImages = []
    for i in range(len(Images)):
        ResizeImages.append(cv2.resize(Images[i], dim, interpolation = cv2.INTER_AREA))
    return ResizeImages

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
	print(output[:, img1.shape[1]:].shape, img1.shape)
	output[:, :img1.shape[1] - w] = img1[:, :img1.shape[1] - w]
	output[:, img1.shape[1]:, :] = img2[:, w:]
	for i in range(w):
		output[:, img1.shape[1] - w + i] = np.floor(img1[:, img1.shape[1] - w + i] * (1 - float(i/w)) + img2[:, i] * (float(i/w)))
	cv2.imwrite("blending.png", output)
	return output

def multi_band_blending(img1, img2, overlap_w):
	def preprocess(img1, img2, overlap_w):

		w1 = img1.shape[1]
		w2 = img2.shape[1]

		shape = np.array(img1.shape)

		shape[1] = w1 + w2 - overlap_w

		subA, subB, mask = np.zeros(shape), np.zeros(shape), np.zeros(shape)
		print("shape", shape, "img1:", img1.shape, "img2", img2.shape, "overlap", overlap_w)
		subA[:, :w1] = img1
		subB[:, w1 - overlap_w:] = img2
		mask[:, :w1 - overlap_w // 2] = 1

		return subA, subB, mask


	def Gaussian(img, leveln):
		GP = [img]
		for i in range(leveln - 1):
		    GP.append(cv2.pyrDown(GP[i]))
		return GP


	def Laplacian(img, leveln):
		LP = []
		for i in range(leveln - 1):
		    next_img = cv2.pyrDown(img)
		    # print(img.shape, cv2.pyrUp(next_img, img.shape[1::-1]).shape)
		    LP.append(img - cv2.pyrUp(next_img, dstsize = img.shape[1::-1]))
		    img = next_img
		LP.append(img)
		return LP


	def blend_pyramid(LPA, LPB, MP):
		blended = []
		for i, M in enumerate(MP):
			blended.append(LPA[i] * M + LPB[i] * (1.0 - M))
		return blended


	def reconstruct(LS):
	    img = LS[-1]
	    for lev_img in LS[-2::-1]:
	        img = cv2.pyrUp(img, dstsize = lev_img.shape[1::-1])
	        img += lev_img
	    return img

	subA, subB, mask = preprocess(img1, img2, overlap_w)

	leveln = int(np.floor(np.log2(min(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1]))))
   
	print("max level", leveln)
	leveln = 8
	print("level", leveln)
	# Get Gaussian pyramid and Laplacian pyramid
	MP = Gaussian(mask, leveln)
	LPA = Laplacian(subA, leveln)
	LPB = Laplacian(subB, leveln)

    # Blend two Laplacian pyramidspass
	blended = blend_pyramid(LPA, LPB, MP)

	# Reconstruction process
	result = np.clip(reconstruct(blended), 0, 255)
	cv2.imwrite("output.png", result.astype(int))
	return result.astype(int)

def main():
	parser = argparse.ArgumentParser(description='Process some images to do image stitching.')
	parser.add_argument("--file", help="input image feature file name")
	parser.add_argument("--output", help="the output file file name", default="./output/output.jpg")
	args = parser.parse_args()

	imgs, fs = read_img(args.file)
	imgs = DownSampling(imgs, 10)
	#warping(imgs[0], fs[0])
	imgs = warping_imgs(imgs, fs)
	multi_band_blending(imgs[7], imgs[6], 120)
if __name__ == '__main__':
	main()