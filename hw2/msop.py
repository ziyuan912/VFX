import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from tqdm import tqdm as tqdm

def ReadImages(path):
	images = np.array([cv2.imread('{}/wraping{}.jpg'.format(path, i)) for i in range(18)])
	# images = np.array([cv2.imread('../test.jpg')])
	return images

def HarrisCornerDetector(image, windowsize, descriptorwindow, k=0.05):
	features = []
	img = np.copy(image)
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	image_gaussianfilter = cv2.GaussianBlur(gray_image, (windowsize, windowsize), 1)
	Ix = image_gaussianfilter - np.roll(image_gaussianfilter, -1, axis=1)
	Iy = image_gaussianfilter - np.roll(image_gaussianfilter, -1, axis=0)

	Ixx = cv2.GaussianBlur(np.multiply(Ix, Ix), (windowsize, windowsize), 1)
	Iyy = cv2.GaussianBlur(np.multiply(Iy, Iy), (windowsize, windowsize), 1)
	Ixy = cv2.GaussianBlur(np.multiply(Ix, Iy), (windowsize, windowsize), 1)

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

		## descriptor
		descriptors = []
		for x, y in features:
			descriptor = []
			offset = descriptorwindow//2
			for x_ in range(-offset, offset+1):
				for y_ in range(-offset, offset+1):
					if(x + x_ >= images[i].shape[0] or y + y_ >= images[i].shape[1]):
						descriptor.append(0)
					else:
						descriptor.append(images[i][x+x_, y+y_])
			descriptors.append(np.asarray(descriptor))

		all_features.append(features)
		all_descriptors.append(np.asarray(descriptors))

	return np.asarray(all_features), np.asarray(all_descriptors)

def FeatureMatching(images, features, descriptors, match_distance=3000):
	all_matching_pairs = []
	for i in range(len(images)):
		f1 = features[i]
		f2 = features[i+1] if i+1 != len(images) else features[0]
		d1 = descriptors[i]
		d2 = descriptors[i+1] if i+1 != len(images) else descriptors[0]
		matching_pairs = []
		for j in tqdm(range(len(d1))):
			dif = []
			if(f1[j][1] > images[0].shape[1]/2):
				continue
			for k in range(len(d2)):
				dif.append(np.sum(np.absolute(d1[j]-d2[k])))
			macth_point = np.argsort(dif)[0]
			second_match_point = np.argsort(dif)[1]
			if(dif[macth_point] < match_distance 
				and dif[second_match_point] - dif[macth_point] > 200
				and f2[macth_point][1] > images[0].shape[1]/2
				and np.absolute(f2[macth_point][0] - f1[j][0]) < 20):
				matching_pairs.append(np.array([f1[j], f2[macth_point]]))
		matching_pairs = np.asarray(matching_pairs)
		all_matching_pairs.append(matching_pairs)
		print('pic{}, match pairs:{}'.format(i, matching_pairs.shape[0]))
	return np.asarray(all_matching_pairs)

def ImageMatching(images, matching_pairs):
	match_image = images[0]
	f = open('match.txt', 'w')
	for i in range(len(images)):
		xdis = images[0].shape[1] + matching_pairs[i][:, 0, 1] - matching_pairs[i][:, 1,1]
		# for j in range(len(matching_pairs)):
		# 	xdis.append(images[0].shape[1] + matching_pairs[j][0][1] - matching_pairs[j][1][1])
		xdis = int(np.mean(xdis))
		nextimage = images[i+1] if i+1 != len(images) else images[0]
		# match_image[:, :xdis+1] = (match_image[:, :xdis+1] + nextimage[:, -xdis:]) / 2
		match_image = np.concatenate((nextimage[:, :-xdis//2], match_image[:, xdis//2:]), 1)
		f.write(str(i) + ',' + str(xdis//2) + '\n')
		cv2.imwrite('{}/matchimg{}.jpg'.format('../output', i), np.concatenate((nextimage[:, :-xdis//2], images[i][:, xdis//2:]), 1))
	f.close()


	return match_image

def main():
	parser = argparse.ArgumentParser(description="Use Multi-Scale Oriented Patches to find features of images.")
	parser.add_argument("-p", "--path", help="input path name of images")
	parser.add_argument("-o", "--output", help="ouput path name of images", default='./output')
	# parser.add_argument("-f", "--featurenum", help="number of features", default=250)
	parser.add_argument("-w", "--windowsize", help="window size", default=9)
	parser.add_argument("-d", "--descriptorwindow", help="descriptor window size", default=5)	
	args = parser.parse_args()
	
	images = ReadImages(args.path)

	features, descriptors = MultiScaleHarrisCornerDetector(images, args.windowsize, args.descriptorwindow)
	# np.save('features', features)
	# np.save('descriptors', descriptors)


	for i, img in enumerate(images):
		image = np.copy(img)
		for feature in features[i]:
			cv2.circle(image, (feature[1], feature[0]), 1, (0, 0, 255), -1)
		cv2.imwrite('{}/img{}.jpg'.format(args.output, i), image)

	matching_pairs = FeatureMatching(images, features, descriptors)
	np.save('matching_pairs', matching_pairs)
	# matching_pairs = np.load('matching_pairs.npy')

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

	# match_image = ImageMatching(images, matching_pairs)
	# cv2.imwrite('{}/matchimg.jpg'.format(args.output), match_image)


if __name__ == '__main__':
	main()