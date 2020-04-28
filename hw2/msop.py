import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from tqdm import tqdm as tqdm

def DownSampling(image, scale_percent=50):
	"""
	Resize the Images
	"""
	width = int(image.shape[1] * scale_percent / 100)
	height = int(image.shape[0] * scale_percent / 100)
	return cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)

def ReadImages(path):
	images = np.array([cv2.imread('{}/prtn{}.jpg'.format(path, i)) for i in range(18)])
	# images = np.array([cv2.imread('../test.jpg')])
	return images

def HarrisCornerDetector(image, windowsize, k=0.04):
	features = []
	img = np.copy(image)
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	image_gaussianfilter = cv2.GaussianBlur(gray_image, (7, 7), 1)
	Ix = image_gaussianfilter - np.roll(image_gaussianfilter, -1, axis=1)
	Iy = image_gaussianfilter - np.roll(image_gaussianfilter, -1, axis=0)

	Ixx = cv2.GaussianBlur(np.multiply(Ix, Ix), (7, 7), 1)
	Iyy = cv2.GaussianBlur(np.multiply(Iy, Iy), (7, 7), 1)
	Ixy = cv2.GaussianBlur(np.multiply(Ix, Iy), (7, 7), 1)

	det = Ixx*Iyy - Ixy**2
	trace = Ixx+Iyy

	## distance
	# R = det - k*trace**2

	# candidates = []
	# for i in range(image.shape[0]):
	# 	for j in range(image.shape[1]):
	# 		candidates.append([i, j, R[i][j]])

	# candidates.sort(key=lambda x:x[2], reverse=True)
	# features.append(candidates.pop(0)[:2])

	# while(len(features) < featurenum and candidates):
	# 	candidate = candidates.pop(0)
	# 	flag = 1
	# 	for feature in features:
	# 		xdis = feature[1]-candidate[1]
	# 		ydis = feature[0]-candidate[0]
	# 		if(xdis**2+ydis**2 <= windowsize**2):
	# 			flag = 0
	# 			break
	# 	if(flag):
	# 		features.append(candidate[:2])

	## window size
	offset = windowsize // 2
	R = det - k*trace**2
	R[:offset, :] = 0
	R[-offset:, :] = 0
	R[:, :offset] = 0
	R[:, -offset:] = 0


	for i in range(R.shape[0]):
		for j in range(R.shape[1]):
			if((i >= offset and i < R.shape[0]-offset) and (j >= offset and j < R.shape[1]-offset) and R[i][j] != 0):
				newR = np.zeros((windowsize, windowsize))
				now_window = R[i-offset:i+offset+1, j-offset:j+offset+1]
				max_index_col,max_index_row = np.unravel_index(now_window.argmax(), now_window.shape)
				newR[max_index_col, max_index_row] = now_window[max_index_col, max_index_row]
				R[i-offset:i+offset+1, j-offset:j+offset+1] = newR
				# if((R[i][j] != np.max(R[i-offset:i+offset+1, j-offset:j+offset+1])) or R[i][j] <= 10):
				# 	R[i][j] = 0

	for i in range(R.shape[0]):
		for j in range(R.shape[1]):
			if(R[i][j] != 0):
				features.append([i, j])

	return np.asarray(features)

def MOSPDescriptor(images, features):
	descriptor_vectors = []

	return descriptor_vectors

def MultiScaleHarrisCornerDetector(images, windowsize, level=3):
	all_features = []
	for i in tqdm(range(len(images))):
		pyramid_img = np.copy(images[i])
		pyramid_imgs = [pyramid_img]
		for k in range(level-1):
			pyramid_img = DownSampling(cv2.GaussianBlur(pyramid_img, (7, 7), 1))
			pyramid_imgs.append(pyramid_img)

		features = []
		for k in range(level):
			features.append(HarrisCornerDetector(pyramid_imgs[level-k-1], windowsize)*2**(level-k-1))
		features = np.vstack((arr for arr in features))
		features = np.unique(features, axis=0)
		all_features.append(features)

	all_features = np.asarray(all_features)

	return all_features

def main():
	parser = argparse.ArgumentParser(description="Use Multi-Scale Oriented Patches to find features of images.")
	parser.add_argument("-p", "--path", help="input path name of images")
	parser.add_argument("-o", "--ouput", help="ouput path name of images", default='../output')
	# parser.add_argument("-f", "--featurenum", help="number of features", default=250)
	parser.add_argument("-w", "--windowsize", help="window size", default=21)
	args = parser.parse_args()
	
	images = ReadImages(args.path)

	features = MultiScaleHarrisCornerDetector(images, args.windowsize)

	for i, img in enumerate(images):
		image = np.copy(img)
		for feature in features[i]:
			cv2.circle(image, (feature[1], feature[0]), 1, (0, 0, 255), -1)
		cv2.imwrite('{}/img{}.jpg'.format(args.ouput, i), image)

if __name__ == '__main__':
	main()