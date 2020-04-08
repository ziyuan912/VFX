# VFX HW1
 B05902050 黃子源
 B05902106 宋昶松
 ## Introduction
In this project, we inplemented image alignment, hdr transform, and tonemapping to make pictures with different shutter time become more clear.

To reproduce our result, simply run:
`python3 HDR.py --file images_info.txt`

`image_ifo.txt` is the file that specifeis the images we want to read and the shutter time of every image. You are welcome to use other arguments to adjust some parameters or do hdr in other image sets. You can check the arguements by running `python3 HDR.py --help` or reading the following section.
### Arguments
* **file:** the txt file which specify the images we want to read and the shutter time of every image.
* **l:** the amount of smoothness of hdr function, default = 100.
* **filter_size:** the size of the gaussion filter we inplement in the tonemapping.
* **downsample:** downsample the input image into specified scale to speed up the calculation, default = 1.0.
* **compression:** the compression factor of tone mapping, default = 0.8.
* **output:** the output file name to store the final result.

The txt image info file is written in the following format:
```
# Number of Images
10
# Filename              1/shutter_speed
image file path1         shutter speed 1
image file path2         shutter speed 2
.                        .
.                        .
.                        .
image file path10        shutter speed 10
```
## Image Alignment
We implemented the image alignment by Median Threshold Bitmap(MTB) alignment. The implementation steps are shown as follows.

### Preprocessing
First, we converted images into greyscale by $Y = (54*R + 183*G + 19*B)/256$.

Second, we constructed bitmap by thresholding the input greyscale image using the median of intensities and exclusion maps by thresholding the input image using 40% and 60% of intensities.

### Search for the optimal offset
We compared images based on the middle image. Then, scale by 2 when passing down and try its 9 neighbors to find the minimum difference.
The difference is calculated by three steps: XOR with bitmap, AND with exclusion maps, Bit counting by table lookup.

## HDR 
We implemented HDR function by the way shown in class. The procedure is mostly the same. We only changed few details to make the output look better.
### Pixel Sampling
To make the response curve smooth and stable in every experiment, we sampled points in every different intensity of the median image. That is, the number of sample $N = 256$, by finding where each pixel value appears in median image, and store the pixel value of the same positon in other images. 
### Recovered Response Curve
The weight function $W$ is defined as follow:
$$W(z) = 
\left\{  
             \begin{array}{**lr**}  
             z - Z_{min} & for & z \leq 1/2(Z_{min} + Z_{max})\\  
             Z_{max} - z & for & z > 1/2(Z_{min} + Z_{max})\\       
             \end{array}  
\right.    
$$
The weight function is used to reduce the effect of the extreme pixel value. Then we constructed optimazation matrix $Ax = b$ traught in class. The optimized matrix x can be calculated by `np.linalg.lstsq(A, b, rcond=None)`. The response function is mapped to the top 256 value of matrix x. The response curve of RGB is shown below.
### Reconstruct Radiance Map
The function we used to reconstruct radiance map is defined as follow:
$lnE_i = \dfrac{\sum^P_{j=1}w(Z_{ij})(g(Z_{ij}) - ln\Delta t_j)}{\sum^P_{j=1}w(Z_{ij})}$

In order to reduce noise, we combined pixels by weighted mean and got a more reliable estimation
## Tone Mapping
We used bilateral tonemapping to map the hdr image to 0-255 pixel value. The implementation steps are shown below.
### Intensity Calculation
We devided the hdr image into intensity field and color field. 

$IntensityFunction = 0.2126 * R + 0.7152 * G + 0.0722 * B$
$ColorFunction =PixelValue / Intensity$

After the calculation, we take the log of the intensity field to reduce the difference between adjacent pixels.
### Bilateral Filter
After doing intensity calculation, we apply the 5x5 bilateral filter on log intensity to calcalate the large scale intensity. The bilateral filter function is defined below:
$$J(x) = 1/k(x)*\sum_{x_i \in \xi}f(\parallel x_i - x \parallel)g(\parallel I(x_i) - I(x)\parallel)I(x_i)$$
Where
* $J(x)$ is the bilateral filter function.
* $\xi$ is the gaussion domain, which is a 5x5 filter in this experiment.
* $f(x)$ is a gaussion function with $\sigma = 1000$.
* $g(x)$ is a gaussion function with $\sigma = 2000$.
* $I(x)$ is the log intensity value of point $x$.
* $k(x) = \sum_{x_i \in \xi}f(\parallel x_i - x \parallel)g(\parallel I(x_i) - I(x)\parallel)$

The large scale intensity can generate detail intensity field by log intensity subtract large scale intensity. The detail intensity field can retain the detail of the original image, and will not let the filter blur them.

### Reduce Contrast
We multiplied 0.8 by the large scale intensity in order to compress large scale feature. After that, added detail intensity back to it to create the new log intensity image. We have tried the compression number from 0.5 to 1.0, and found that 0.7 has the best output image.

To get the tone mapping image, we recovered the log intensity field by exponential, and merge the intensity field with color field by multiplying them together.

### Normalization
After getting the tonemapping output, we normalized the output into 0 to 255. However, the normalized output is still too dark since the distribution of the pixel is still not uniform, that is, the mean value of all pixels are too small. So we rescaled the picture by the original pictures' mean.
$N(x) = x * Mean(OriginImages)/Mean(Output)$
Where
* $N(x)$ is the normalization function of $x$.
* $Mean(imgs)$ is the weighted average of the original images. The weight function $W$ is the same as the one used in HDR.

## Experiment

### Rescale for Normalization
![](https://i.imgur.com/PDCxJdP.jpg)

### Compression Factor for Contrast Reducing
![](https://i.imgur.com/r6Lqd9r.jpg)


## Result
The result is stored in `output/output.jpg`, and the input image is stored under `images/`.
### Input
### Response Curve
### Radiance Map
### Output