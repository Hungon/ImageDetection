import cv2
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    img = cv2.imread('sample_images/watch.jpg',cv2.IMREAD_COLOR)
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.plot([200,300,400],[100,200,300],'c', linewidth=5)
    plt.show()
    cv2.imwrite('saved_images/watchgray.png', img)
