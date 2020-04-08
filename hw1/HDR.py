import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import random
import cv2
from PIL import Image
from tqdm import tqdm as tqdm


Zmin = 0
Zmax = 255
n = 256
N = 256
def read_imgs(file):
	imgs = list()
	P = 0
	B = []
	f = open(file)
	lines = f.readlines()
	f.close()
	for i in range(len(lines)):
		if "# Number of Images" in lines[i]:
			P = int(lines[i+1])
			i += 1
		elif "# Filename  1/shutter_speed" in lines[i]:
			i += 1
			while i < len(lines):
				feature = lines[i].split()
				imgs.append(cv2.imread(feature[0]))
				B.append(np.log(1/float(feature[1])))
				i += 1
	return imgs, B, P

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

def Shift(bitmap1, bitmap2, exclusionMaps1, exclusionMaps2, scale):
	if(scale < 0):
		return [0, 0]

	b1, b2, eb1, eb2 = DownSampling([bitmap1, bitmap2, exclusionMaps1, exclusionMaps2], 50)
	xs, ys = Shift(b1, b2, eb1, eb2, scale-1)
	xs *= 2
	ys *= 2

	shift = [0, 0]
	min_err = bitmap1.shape[0] * bitmap2.shape[1]

	for x in [-1, 0, 1]:
		for y in [-1, 0, 1]:
			M = np.float32([[1, 0, xs+x],[0, 1, ys+y]])
			shift_b2 = cv2.warpAffine(bitmap2 ,M, (bitmap2.shape[1], bitmap2.shape[0]))
			shift_eb2 = cv2.warpAffine(exclusionMaps2 ,M, (bitmap2.shape[1], bitmap2.shape[0]))

			diff_b = np.logical_xor(bitmap1, shift_b2)
			diff_b = np.logical_and(diff_b, exclusionMaps1)
			diff_b = np.logical_and(diff_b, shift_eb2)

			err = np.count_nonzero(diff_b)
			if(err < min_err):
				shift = [x, y]
				min_err = err
	shift[0] += xs
	shift[1] += ys
	return shift

def MTBAlign(Images, scale=5):
	"""
	Align images to the same position.
	"""
	P = len(Images)
	Size = [Images[0].shape[0], Images[0].shape[1]]
	alignImages = []

	greyImages = np.zeros((P, Size[0], Size[1]))
	for i in range(P):
		b, g, r = cv2.split(Images[i])
		r = np.array(r, dtype=np.float32)
		g = np.array(g, dtype=np.float32)
		b = np.array(b, dtype=np.float32)
		greyImages[i] = (54 * r + 183 * g + 19 * b)/256

	bitmap = np.zeros((P, Size[0], Size[1]))
	exclusionMaps = np.zeros((P, Size[0], Size[1]))
	for i in range(P):
		t1, median, t2 = np.percentile(greyImages[i], [40, 50, 60])
		bitmap[i] = np.where(greyImages[i] > median, 1, 0)
		exclusionMaps[i] = np.where(np.logical_or(greyImages[i] < t1, greyImages[i] > t2), 1, 0)

	standard = int(P/2)
	print("------------ image alignment ------------")
	for i in tqdm(range(P)):
		if(i == standard):
			alignImages.append(Images[i])
			continue
		xs, ys = Shift(bitmap[standard], bitmap[i], exclusionMaps[standard], exclusionMaps[i], scale)
		M = np.float32([[1, 0, xs],[0, 1, ys]])
		alignImages.append(cv2.warpAffine(Images[i] ,M, (Size[1], Size[0])).astype(np.uint8))

	return alignImages

def get_sample_point(imgs, intensity, median, channel):
	output_row, output_col = np.where(imgs[median][:, :, channel] == intensity)
	if len(output_row) == 0:
		return (random.randint(0, imgs[0].shape[0]-1), random.randint(0, imgs[0].shape[1]-1))
	rnd = random.randrange(len(output_row))
	return (output_row[rnd], output_col[rnd])

def Z_generator(imgs, img_shape, N):
	Z = np.zeros((N, len(imgs), 3))
	for i in range(N):
		for k in range(3):
			sample_point = get_sample_point(imgs, i, len(imgs) // 2, k)
			for j in range(len(imgs)):			
				Z[i, j] = imgs[j][sample_point[0], sample_point[1]]
	Z = Z.astype(int)
	return Z

def W(z):
	if(z == 128):
		return 1.0
	elif(0 <= z < 128):
		return float(z/128)
	elif(128 < z <= 255):
		return float((257 - z)/128)
	else:
		return 0.0

def HDR(A, B, Z, b, l):
	k = 0
	for i in range(Z.shape[0]):
		for j in range(Z.shape[1]):
			wij = W(Z[i,j] + 1)
			A[k, Z[i, j]] = wij
			A[k, n + i] = -1*wij
			b[k, 0] = wij * B[j]
			k += 1
	A[k, 128] = 1
	k += 1
	for i in range(n - 2):
		A[k, i] =  l * W(i + 1)
		A[k, i + 1] = -2 * l * W(i + 1)
		A[k, i + 2] = l * W(i + 1)
		k += 1
	x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
	return x

def gaussian(x, sig):
    return np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.))) * (1.0 / (2 * np.pi * (sig ** 2)))

def bilateral_filter(log_intensity, x, y, args):
	k = 0
	output = 0
	sigma_g = 2000
	sigma_f = 1000
	filter_size = args.filter_size
	for i in range(filter_size):
		for j in range(filter_size):
			movei = i - filter_size // 2
			movej = j - filter_size // 2
			if x + movei >= log_intensity.shape[0] or x + movei < 0 or y + movej >= log_intensity.shape[1] or y + movej < 0:
				continue
			f = gaussian(np.sqrt(movei ** 2 + movej ** 2), sigma_f)
			g = gaussian(log_intensity[x][y] - log_intensity[x + movei][y + movej], sigma_g)
			k += f * g
			output += f * g * log_intensity[x + movei][y + movej]
	output = output/k
	return output

def bilateral(img, args):
	intensity = 0.0722*img[:, :, 0] + 0.7152*img[:, :, 1] + 0.2126*img[:, :, 2]
	log_intensity = np.log(intensity)
	log_large_scale = np.zeros(log_intensity.shape)
	print("------------ tone mapping ------------")
	for i in tqdm(range(log_large_scale.shape[0])):
		for j in range(log_large_scale.shape[1]):
			log_large_scale[i, j] = bilateral_filter(log_intensity, i, j, args)
	log_detail = log_intensity - log_large_scale
	log_output = log_large_scale * args.compression + log_detail
	output = np.zeros((img.shape[0], img.shape[1], 3))
	output[:, :, 0] = img[:, :, 0]/intensity * np.exp(log_output)
	output[:, :, 1] = img[:, :, 1]/intensity * np.exp(log_output)
	output[:, :, 2] = img[:, :, 2]/intensity * np.exp(log_output)
	return output


def build_HDR_image(imgs, g, t_delta):
	HDR_output = np.zeros((imgs[0].shape[0], imgs[0].shape[1], 3), dtype='float32')
	radiance_map = np.zeros((imgs[0].shape[0], imgs[0].shape[1], 3), dtype='float32')
	print("------------ building HDR image ------------")
	for i in tqdm(range(imgs[0].shape[0])):
		for j in range(imgs[0].shape[1]):
			for k in range(3):
				eup = 0
				edown = 0
				for p in range(len(imgs)):
					eup += W(imgs[p][i, j, k] + 1)*(g[imgs[p][i, j, k], k] - t_delta[p])
					edown += W(imgs[p][i, j, k] + 1)
				radiance_map[i, j, k] = float(eup/edown)
				HDR_output[i, j, k] = np.exp(float(eup/edown))
	plt.imshow(radiance_map[:, :, 0], origin="lower", cmap='rainbow', interpolation='nearest')
	plt.gca().invert_yaxis()
	plt.colorbar()
	plt.title("Radiance Map")
	plt.savefig("output/radiance_map.png", dpi = 300)
	return HDR_output

def output_rescale(image, origin_image):
    b, g, r = cv2.split(image)
    b2, g2, r2 = cv2.split(origin_image)
    b *= np.average(b2) / np.nanmean(b)
    g *= np.average(g2) / np.nanmean(g)
    r *= np.average(r2) / np.nanmean(r)
    image = cv2.merge((b,g,r))
    return image

def output_rescale_all(image, origin_images):
	b, g, r = cv2.split(image)
	mean_image = np.zeros((image.shape[0], image.shape[1], 3))
	print("------------ rescaling output ------------")
	for i in tqdm(range(image.shape[0])):
		for j in range(image.shape[1]):
			for k in range(3):
				wsum = 0
				pixelvalue = 0
				for l in range(len(origin_images)):
					w = W(origin_images[l][i, j, k])
					pixelvalue += w * origin_images[l][i, j, k]
					wsum += w
				mean_image[i, j, k] = pixelvalue / wsum
	b2, g2, r2 = cv2.split(mean_image)
	b *= np.average(b2) / np.nanmean(b)
	g *= np.average(g2) / np.nanmean(g)
	r *= np.average(r2) / np.nanmean(r)
	image = cv2.merge((b,g,r))
	return image




def main():
	parser = argparse.ArgumentParser(description='Process some images to do HDR.')
	parser.add_argument("--file", help="input image feature file name")
	parser.add_argument("--downsample", help="downsample the input image into specified scale to speed up the calculation", type=float, default = 10)
	parser.add_argument("--l", help="determine the amount of smoothness", type=float, default=100)
	parser.add_argument("--filter_size", help="determine the gaussion filter size", type=int, default=5)
	parser.add_argument("--compression", help="the compression factor of tone mapping", type=float, default=0.8)
	parser.add_argument("--output", help="the output file file name", default="./output/output.jpg")
	args = parser.parse_args()

	imgs, B, P = read_imgs(args.file)
	imgs = DownSampling(imgs, args.downsample)
	imgs = MTBAlign(imgs, 3)
	
	HDR_output = np.zeros((imgs[0].shape[0], imgs[0].shape[1], 3), dtype='float32')
	l = args.l

	Z = Z_generator(imgs, imgs[0].shape, N)
	A = np.zeros((Z.shape[0] * Z.shape[1] + n + 1, n + Z.shape[0], 3) )
	B = np.array(B)
	b = np.zeros((A.shape[0], 1, 3))

	g = np.zeros((256, 3))

	for i in range(3):
		x = HDR(A[:, :, i], B, Z[:, :, i], b[:, :, i], l)
		g[:, i] = x[: n].reshape(n)
		lE = x[n: x.shape[0]]

	# plot response curve
	pixel_range = np.arange(256)
	plot1 = plt.plot(pixel_range, g[:, 0], 'r')
	plot2 = plt.plot(pixel_range, g[:, 1], 'g')
	plot3 = plt.plot(pixel_range, g[:, 2], 'b')
	plt.title("response curve")
	plt.savefig("./output/response_curve.png")
	plt.close('all')

	HDR_output = build_HDR_image(imgs, g, B)
	output = bilateral(HDR_output, args)
	for i in range(3):
		output[:, :, i] = cv2.normalize(output[:, :, i], np.array([]), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
	outputimg = output_rescale_all(output, imgs)
	cv2.imwrite(args.output, outputimg)



if __name__ == '__main__':
	main()