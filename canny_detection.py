import numpy as np
import matplotlib.pyplot as plt
import cv2

""""
lookup table**
At the same time, we also know that the value of each pixel only lies between 0 and 255 (inclusive). What if we can
?pre-compute and store the transformed values s=T(r) for values s=T(r) for r∈{0,1,2,...,255}, that is,
for all possible values of the input. If we do so, then irrespective of the dimensions (number of pixels) in our
input image, we will never need more than 256 computations. So, using this strategy, we traverse our matrix and do
a simple lookup of the pre-computed values. This is called a lookup table approach ...

So, what is color quantization?
Color quantization is the process of reducing the number of distinct colors in an image.
Normally, the intent is to preserve the color appearance of the image as much as possible, while reducing the number of
colors, whether for memory limitations or compression.
https://www.youtube.com/watch?v=O0suOccKLbs&ab_channel=MisterCode
"""

"""
Convolution
 mask  Put simply; a mask allows us to focus only on the portions of the image that interests us.
 filter - bloor or sharpen img
 https://learnopencv.com/image-filtering-using-convolution-in-opencv/#intro-convo-kernels
 """
#
image = cv2.imread('photo/d.jpeg')


kernel1 = np.array([[0.1, 0.1, 0.1],
                    [0.1, 1, 0.1],
                    [0.1, 0.1, 0.1]])

sobel = np.array([[1,0,-1],
                 [2,0,-2],
                 [1,0,-1]])
op_sobel = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])

sobel_img = cv2.filter2D(image,-1,sobel)
op_sobel_img = cv2.filter2D(image,-1,op_sobel)

kernel2 = np.ones((6, 6), np.float32) / 30

median = cv2.medianBlur(src=image, ksize=5)

after = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)

kernel3 = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])

sharp_img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel3)


cv2.imshow('Original', image)
# cv2.imshow('Median Blurred', median)
# cv2.imshow('after', after)
cv2.imshow('Sharpened', sharp_img)
# cv2.imshow('Sobel', sobel_img)
# cv2.imshow('op_Sobel', op_sobel_img)

cv2.imwrite('after.jpg', after)
cv2.waitKey()
cv2.destroyAllWindows()


"""
thresh hold 
"""
# img = cv2.imread('photo/a.jpg', 0)
# plt.imshow(img, cmap='gray')
# plt.show()
#
# ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
# plt.imshow(thresh1, cmap='gray')
# plt.show()

"""
edge detection 
https://learnopencv.com/edge-detection-using-opencv/
Sobel Edge Detection is one of the most widely used algorithms for edge detection. The Sobel Operator detects edges that
are marked by sudden changes in pixel intensity, as shown in the figure below.
 
Edges in images are areas with strong intensity contrasts; a jump in
intensity from one pixel to the next.
The process of edge detection significantly reduces the amount of data
and filters out unneeded information, while preserving the important
structural properties of an image.
There are many different edge detection methods, the majority of which
can be grouped into two categories:
->Gradient,
->Laplacian.
The gradient method detects the edges by looking for the maximum
and minimum in the first derivative of the image.
The Laplacian method searches for zero crossings in the second
derivative of the image .

[-1,0,1]
[-2,0,2] -> that kernel wll help us to find the vertical changes 
[-1,0,1]

[1, 2, 1]
[0, 0, 0] -> that kernel wll help us to find the horizontal  changes  
[-1-2,-1]


# """
# img = cv2.imread('photo/tiger.jpeg', 0)
# img_blur = cv2.GaussianBlur(img, (3, 3), 0, 0)
# # plt.imshow(img_blur,cmap='gray')
# # plt.show()
# sobol = cv2.Sobel(img_blur, -1, 1, 1)
# plt.imshow(sobol,cmap='gray')
# plt.show()

"""
canny edges detection algorithm
https://www.pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/
https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123

1. blur the img
2. compute the Gradient - x , Gradient - y
3. compute the magnitude is used to measure how strong the change in image intensity is.
4. compute the gradient orientation is used to determine in which direction the change in intensity is pointing
5. quantization the gradient angle to 4 option 
6. delete all pix that not local max 
7. thresholding
"""


def canny_detection(img):
    # upload the photo ' and blur it
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 1.4)
    # get the horizontal and vertical gradient
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)
    #  get the magnitude and the orientation
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    # thresholding
    mag_max = np.max(mag)
    weak_th = mag_max * 0.1
    strong_th = mag_max * 0.8

    height, width = img.shape

    for x in range(width):
        for y in range(height):
            grad_ang = ang[y, x]
            grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

            if grad_ang <= 22.5:
                neighb_1_x, neighb_1_y = x - 1, y
                neighb_2_x, neighb_2_y = x + 1, y

            # top right (diagonal-1) direction
            elif 22.5 < grad_ang <= (22.5 + 45):
                neighb_1_x, neighb_1_y = x - 1, y - 1
                neighb_2_x, neighb_2_y = x + 1, y + 1

            # In y-axis direction
            elif (22.5 + 45) < grad_ang <= (22.5 + 90):
                neighb_1_x, neighb_1_y = x, y - 1
                neighb_2_x, neighb_2_y = x, y + 1

            # top left (diagonal-2) direction
            elif (22.5 + 90) < grad_ang <= (22.5 + 135):
                neighb_1_x, neighb_1_y = x - 1, y + 1
                neighb_2_x, neighb_2_y = x + 1, y - 1

            # Now it restarts the cycle
            elif (22.5 + 135) < grad_ang <= (22.5 + 180):
                neighb_1_x, neighb_1_y = x - 1, y
                neighb_2_x, neighb_2_y = x + 1, y

            # Non-maximum suppression step
            if width > neighb_1_x >= 0 and height > neighb_1_y >= 0:
                if mag[y, x] < mag[neighb_1_y, neighb_1_x]:
                    mag[y, x] = 0
                    continue

            if width > neighb_2_x >= 0 and height > neighb_2_y >= 0:
                if mag[y, x] < mag[neighb_2_y, neighb_2_x]:
                    mag[y, x] = 0
            # 0if we want more thresh holding ....
            weak_ids = np.zeros_like(img)
            strong_ids = np.zeros_like(img)
            ids = np.zeros_like(img)

            # double thresholding step
            for x in range(width):
                for y in range(height):

                    grad_mag = mag[y, x]

                    if grad_mag < weak_th:
                        mag[y, x] = 0
                    elif strong_th > grad_mag >= weak_th:
                        ids[y, x] = 1
                    else:
                        ids[y, x] = 2
            return mag

#
img = cv2.imread('photo/tiger.jpeg')
after = canny_detection(img)
plt.imshow(after, cmap='gray')
plt.show()
