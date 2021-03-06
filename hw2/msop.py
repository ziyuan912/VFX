import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from tqdm import tqdm as tqdm
from random import randint
import math

def ReadImages(path):
	images = np.array([cv2.imread('{}/wraping{}.jpg'.format(path, i)) for i in range(19)])
	# images = np.array([cv2.imread('../test.jpg')])
	return images

def HarrisCornerDetector(image, windowsize, descriptorwindow, k=0.04):
	features = []
	img = np.copy(image)
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	sigma = 3
	imx = np.zeros(gray_image.shape)
	imy = np.zeros(gray_image.shape)
	filters.gaussian_filter(gray_image, (sigma, sigma), (0, 1), imx)
	filters.gaussian_filter(gray_image, (sigma, sigma), (1, 0), imy)

	# compute components of the Harris matrix
	Ixx = filters.gaussian_filter(imx*imx,sigma)
	Ixy = filters.gaussian_filter(imx*imy,sigma)
	Iyy = filters.gaussian_filter(imy*imy,sigma)

	det = Ixx*Iyy - Ixy**2
	trace = Ixx + Iyy
	R = det - k * trace**2

	# image_gaussianfilter = cv2.GaussianBlur(gray_image, (windowsize, windowsize), 1)
	# Ix = image_gaussianfilter - np.roll(image_gaussianfilter, -1, axis=1)
	# Iy = image_gaussianfilter - np.roll(image_gaussianfilter, -1, axis=0)

	# Ixx = cv2.GaussianBlur(np.multiply(Ix, Ix), (windowsize, windowsize), 1)
	# Iyy = cv2.GaussianBlur(np.multiply(Iy, Iy), (windowsize, windowsize), 1)
	# Ixy = cv2.GaussianBlur(np.multiply(Ix, Iy), (windowsize, windowsize), 1)

	# det = Ixx*Iyy - Ixy**2
	# trace = Ixx+Iyy
	# R = det - k*trace**2

	## window size
	offset = windowsize // 2
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

def MultiScaleHarrisCornerDetector(images, windowsize, descriptorwindow, level=5):
	all_features = []
	all_descriptors = []
	for i in tqdm(range(len(images))):
		pyramid_img = np.copy(images[i])
		pyramid_imgs = [pyramid_img]
		for k in range(level-1):
			pyramid_img = cv2.pyrDown(pyramid_img)
			pyramid_imgs.append(pyramid_img)

		features = []
		for k in range(level):
			features.append(HarrisCornerDetector(pyramid_imgs[level-k-1], windowsize, descriptorwindow)*2**(level-k-1))
		features = np.vstack((arr for arr in features))
		features = np.unique(features, axis=0)

		# ## use cv
		# gray = cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)

		# gray = np.float32(gray)
		# dst = cv2.cornerHarris(gray,2,3,0.04)

		# #result is dilated for marking the corners, not important
		# dst = cv2.dilate(dst,None)

		# # Threshold for an optimal value, it may vary depending on the image.
		# for i, j in gray[dst>0.01*dst.max()]:
		# 	features.append(np.array([i, j]))
		# features = np.array(features)

		## descriptor
		descriptors = []
		img = np.copy(images[i])
		# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# image_gaussianfilter = cv2.GaussianBlur(gray_image, (5, 5), 4.5)
		# Ix = image_gaussianfilter - np.roll(image_gaussianfilter, -1, axis=1)
		# Iy = image_gaussianfilter - np.roll(image_gaussianfilter, -1, axis=0)
		# angle = np.arccos(Ix / np.sqrt(Ix**2+Iy**2)) * 180 / math.pi
		img = (img-np.mean(img))/np.std(img)
		for x, y in features:
			# M = cv2.getRotationMatrix2D((y, x), angle[x][y], 1)
			# rotatedimg = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
			descriptor = []
			offset = descriptorwindow//2
			for x_ in range(-offset, offset+1):
				for y_ in range(-offset, offset+1):
					if(x + x_ >= images[i].shape[0] or y + y_ >= images[i].shape[1]):
						descriptor.append([0, 0, 0])
					else:
						descriptor.append(img[x+x_, y+y_])
			descriptors.append(np.asarray(descriptor))

		all_features.append(features)
		all_descriptors.append(np.asarray(descriptors))

	return np.asarray(all_features), np.asarray(all_descriptors)

def FeatureMatching(images, features, descriptors, match_distance=50):
	all_matching_pairs = []
	for i in range(len(images)):
		f1 = features[i]
		f2 = features[i+1] if i+1 != len(images) else features[0]
		d1 = descriptors[i]
		d2 = descriptors[i+1] if i+1 != len(images) else descriptors[0]
		matching_pairs = []
		for j in tqdm(range(len(d1)-1)):
			if(f1[j][1] > images[0].shape[1]/2):
			 	continue

			d1_ = np.concatenate([d1[j] for n in range(d2.shape[0])]).reshape((d2.shape[0], 25, 3))
			dif = np.sum(np.sum(np.absolute(d1_ - d2), axis=1), axis=1)
			"""d1_ = np.concatenate([d1[j] for n in range(d2.shape[0])]).reshape((d2.shape[0], 25, 3))
			dif = np.sum(np.sum((d1_ - d2)**2, axis=1), axis=1)"""
			dif = np.array(dif, dtype=np.float32)
			macth_point = np.argsort(dif)[0]
			second_match_point = np.argsort(dif)[1]

			# if(dif[macth_point] < match_distance 
			# 	and dif[second_match_point] - dif[macth_point] > 1
			# 	and f2[macth_point][1] > images[0].shape[1]/2
			# 	and np.absolute(f2[macth_point][0] - f1[j][0]) < 20):
			# 	matching_pairs.append(np.array([f1[j], f2[macth_point]]))

			is_match = dif[macth_point] == 0 or (dif[macth_point]/dif[second_match_point] < 0.9
				and f2[macth_point][1] > images[0].shape[1]/2
				and np.absolute(f2[macth_point][0] - f1[j][0]) < 10)
			if(is_match):
				matching_pairs.append(np.array([f1[j], f2[macth_point]]))
		matching_pairs = np.asarray(matching_pairs)
		all_matching_pairs.append(matching_pairs)
		print('pic{}, match pairs:{}'.format(i, matching_pairs.shape[0]))
	return np.asarray(all_matching_pairs)

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
   
	# print("max level", leveln)
	leveln = 8
	# print("level", leveln)
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

def ImageMatching(images, matching_pairs):
	padding_images = np.array([cv2.copyMakeBorder(images[i], 70, 70, 0, 0, cv2.BORDER_CONSTANT, value=0) for i in range(images.shape[0])])
	match_image = padding_images[0]
	for i in range(len(images) - 1):
		x_allshift = matching_pairs[i][:, 0, 1] - matching_pairs[i][:, 1, 1] + images[i].shape[1]
		y_allshift = matching_pairs[i][:, 0, 0] - matching_pairs[i][:, 1, 0]
		similar_max = 0
		x_shift = 0
		y_shift = 0
		for shift_idx in range(len(x_allshift)):
			similar_shift = np.sum(abs(x_allshift - x_allshift[shift_idx]) < 10)
			if similar_shift > similar_max:
				similar_max = similar_shift
				x_shift = x_allshift[shift_idx]
				y_shift = y_allshift[shift_idx]

		nextimage = padding_images[i+1]
		
		M = np.float32([[1, 0, 0],[0, 1, -y_shift]])
		match_image=np.asarray(match_image, dtype=np.float32)
		match_image = cv2.warpAffine(match_image, M, (match_image.shape[1], match_image.shape[0]))

		match_image = multi_band_blending(nextimage, match_image, x_shift)
		# cv2.imwrite('{}/matchimg{}.jpg'.format('../output', i), np.concatenate((nextimage[:, :-x_shift//2], images[i][:, x_shift//2:]), 1))


	return match_image

def main():
	parser = argparse.ArgumentParser(description="Use Multi-Scale Oriented Patches to find features of images.")
	parser.add_argument("-p", "--path", help="input path name of images")
	parser.add_argument("-o", "--output", help="ouput path name of images", default='./output')
	# parser.add_argument("-f", "--featurenum", help="number of features", default=250)
	parser.add_argument("-w", "--windowsize", help="window size", type = int , default=5)
	parser.add_argument("-d", "--descriptorwindow", help="descriptor window size", type = int ,default=5)	
	args = parser.parse_args()
	
	images = ReadImages(args.path)

	features, descriptors = MultiScaleHarrisCornerDetector(images, args.windowsize, args.descriptorwindow)
	np.save('features', features)
	np.save('descriptors', descriptors)
	features = np.load('features.npy')
	descriptors = np.load('descriptors.npy')


	for i, img in enumerate(images):
		image = np.copy(img)
		for feature in features[i]:
			cv2.circle(image, (feature[1], feature[0]), 1, (0, 0, 255), -1)
		cv2.imwrite('{}/img{}.jpg'.format(args.output, i), image)

	matching_pairs = FeatureMatching(images, features, descriptors)
	np.save('matching_pairs', matching_pairs)
	matching_pairs = np.load('matching_pairs.npy')
	
	for i in range(len(images)):
		img1 = np.copy(images[i])
		img2 = np.copy(images[i+1] if i+1 != len(images) else images[0])
		pic = np.concatenate((img2, img1), 1)
		for matching_pair in matching_pairs[i]:
			m1 = (matching_pair[0][1]+img2.shape[1], matching_pair[0][0])
			m2 = (matching_pair[1][1], matching_pair[1][0])
			cv2.circle(pic, m2, 1, (0, 0, 255), -1)
			cv2.circle(pic, m1, 1, (0, 0, 255), -1)
			cv2.line(pic, m1, m2, (255, 255, 0), 1)
		cv2.imwrite('{}/matchingimg{}.jpg'.format(args.output, i), pic)

	match_image = ImageMatching(images, matching_pairs)
	cv2.imwrite('{}/matchimg3.jpg'.format(args.output), match_image)


if __name__ == '__main__':
	main()