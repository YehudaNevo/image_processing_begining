import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('gray.png',0)
# hist_val = cv2.calcHist([img],channels=[0],mask=None,histSize=[256],ranges=[0,256])

# plt.plot(hist_val)
# plt.show()


# --- normalized img ..--------

def norm_img(img):

    img_arr = np.asarray(img)
    print(img_arr.shape)

    hist = [0 for _ in range(256)]

    for i in range(len(img_arr)):
        for j in range(len(img_arr[0])):
            hist[img_arr[i][j]] += 1
    plt.plot(hist)
    plt.show()

    cumulative_histogram = [0 for _ in hist]
    cumulative_histogram[0] = hist[0]
    for i in range(1, len(hist)):
        cumulative_histogram[i] = cumulative_histogram[i - 1] + hist[i]

    plt.plot(cumulative_histogram)
    plt.show()
    sum_val = cumulative_histogram[-1]

    normalized_hist = [round(num * 255 / sum_val) for num in cumulative_histogram]

    plt.imshow(img_arr, cmap='gray')
    plt.show()
    # loop over the pixel and normalized them
    for i in range(len(img_arr)):
        for j in range(len(img_arr[0])):
            img_arr[i][j] = normalized_hist[img_arr[i][j]]

    plt.plot(normalized_hist)
    plt.show()
    plt.imshow(img_arr, cmap='gray')
    plt.show()

norm_img()