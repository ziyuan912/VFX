import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import random
import cv2
from PIL import Image


Zmin = 0
Zmax = 255
n = 256
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

def Z_generator(imgs, img_shape, N):
	Z = np.zeros((N, len(imgs), 3))
	for i in range(N):
		rand_point = (random.randint(50, img_shape[0]-50), random.randint(50, img_shape[1]-50))
		for j in range(len(imgs)):
			Z[i, j] = imgs[j][rand_point[0], rand_point[1]]
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

def HDR(A, B, Z, b):
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
		A[k, i] = W(i + 1)
		A[k, i + 1] = -2*W(i + 1)
		A[k, i + 2] = W(i + 1)
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
	intensity = 0.2126*img[:, :, 0] + 0.7152*img[:, :, 1] + 0.0722*img[:, :, 2]
	log_intensity = np.log(intensity)
	log_large_scale = np.zeros(log_intensity.shape)
	for i in range(log_large_scale.shape[0]):
		for j in range(log_large_scale.shape[1]):
			log_large_scale[i, j] = bilateral_filter(log_intensity, i, j, args)
	log_detail = log_intensity - log_large_scale
	#plt.imshow(Image.fromarray(np.exp(log_large_scale)))
	#plt.imshow(Image.fromarray(np.exp(log_detail)))
	log_output = log_large_scale * 0.85 + log_detail
	output = np.zeros((img.shape[0], img.shape[1], 3))
	output[:, :, 0] = img[:, :, 0]/intensity * np.exp(log_output)
	output[:, :, 1] = img[:, :, 1]/intensity * np.exp(log_output)
	output[:, :, 2] = img[:, :, 2]/intensity * np.exp(log_output)
	print(output)
	return output


def build_HDR_image(imgs, g, t_delta):
	HDR_output = np.zeros((imgs[0].shape[0], imgs[0].shape[1], 3), dtype='float32')
	for i in range(imgs[0].shape[0]):
		for j in range(imgs[0].shape[1]):
			for k in range(3):
				eup = 0
				edown = 0
				for p in range(len(imgs)):
					eup += W(imgs[p][i, j, k] + 1)*(g[imgs[p][i, j, k], k] - t_delta[p])
					edown += W(imgs[p][i, j, k] + 1)
				HDR_output[i, j, k] = np.exp(float(eup/edown))
	
	cv2.imwrite("output.hdr", HDR_output)
	np.save("hdr.npy", HDR_output)
	return HDR_output

def intensityAdjustment(image, template):
    g, b, r = cv2.split(image)
    tg, tb, tr = cv2.split(template)
    b *= np.average(tb) / np.nanmean(b)
    g *= np.average(tg) / np.nanmean(g)
    r *= np.average(tr) / np.nanmean(r)
    # image = np.average(template) / np.nanmean(image) * image
    image = cv2.merge((g,b,r))
    return image

def main():
	parser = argparse.ArgumentParser(description='Process some images to do HDR.')
	parser.add_argument("--file", help="input image feature file name")
	parser.add_argument("--N", help="number of sample pixel", type=int ,default=256)
	parser.add_argument("--l", help="determine the amount of smoothness", type=float, default=100)
	parser.add_argument("--filter_size", help="determine the gaussion filter size", type=int, default=5)
	parser.add_argument("--hdr_file", default=None)
	args = parser.parse_args()

	imgs, B, P = read_imgs(args.file)
	HDR_output = np.zeros((imgs[0].shape[0], imgs[0].shape[1], 3), dtype='float32')
	if args.hdr_file != None:
		HDR_output = np.load(args.hdr_file)
	else:
		l = args.l

		Z = Z_generator(imgs, imgs[0].shape, args.N)
		A = np.zeros((Z.shape[0] * Z.shape[1] + n + 1, n + Z.shape[0], 3) )
		B = np.array(B)
		b = np.zeros((A.shape[0], 1, 3))

		g = np.zeros((256, 3))

		for i in range(3):
			x = HDR(A[:, :, i], B, Z[:, :, i], b[:, :, i])
			print(x.shape)
			g[:, i] = x[: n].reshape(n)
			"""for j in range(n):
				if g[j, i] < -3:
					g[j, i] = -3"""
			lE = x[n: x.shape[0]]

		pixel_range = np.arange(256)
		plot = plt.plot(pixel_range, g[:, 0])
		#plt.show()

		HDR_output = build_HDR_image(imgs, g, B)
	output = bilateral(HDR_output, args)
	for i in range(3):
		output[:, :, i] = cv2.normalize(output[:, :, i], np.array([]), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
	output = intensityAdjustment(output, imgs[0])
	cv2.imwrite("tonemap.jpg", output)



if __name__ == '__main__':
	main()