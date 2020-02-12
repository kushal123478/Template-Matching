# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 19:03:19 2019

@author: KUSHAL
"""

import cv2
import numpy as np

image = cv2.imread(r"C:\Users\Kushal Patel\Desktop\Courses\Computer Vision\Homework 2\motherboard-gray.png", cv2.IMREAD_GRAYSCALE )
template = cv2.imread(r"C:\Users\Kushal Patel\Desktop\Courses\Computer Vision\Homework 2\template.png", cv2.IMREAD_GRAYSCALE )

image.shape

################################
#noisy - modified from Shubham Pachori on stackoverflow

def noisy(image, noise_type, sigma):
    if noise_type == "gauss":
        row,col = image.shape
        mean = 0
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_type == "s&p":
        row,col = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type =="speckle":
        row,col = image.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)
        noisy = image + image * gauss
        return noisy

corr = np.zeros((11, 6))
loc = []
image_cpy = image.copy()
image_cpy = cv2.cvtColor(image_cpy, cv2.COLOR_GRAY2RGB)
i = 0
for noiselevel in range(11):   
    j = 0
    loc_temp = []
    for sigma in range(6):
        NoisyImage = np.uint8(noisy(image, 'gauss', noiselevel))
        if sigma == 0:
            GaussBlurred = NoisyImage.copy()
        else:
            GaussBlurred = cv2.GaussianBlur(NoisyImage, ksize = (0, 0),  sigmaX = sigma) 

        
        res = cv2.matchTemplate(GaussBlurred ,template, cv2.TM_CCORR_NORMED)
        corr[i,j] = np.max(res)
        loc_temp.append( (int(np.where(res == np.amax(res))[1]), int(np.where(res == np.amax(res))[0])) )
        j += 1
        if i==10 and j==5:
            save_res = res
    loc.append(loc_temp)
    i+=1

loc = np.asarray(loc)

NoisyImage = np.uint8(noisy(image, 'gauss', 10))
GaussianBlurred = cv2.GaussianBlur(NoisyImage, (29, 29), 5 , 5)

cv2.imwrite(r"C:\Users\Kushal Patel\Desktop\Courses\Computer Vision\Homework 2\FinalResults\Noisy.png", NoisyImage)
cv2.imwrite(r"C:\Users\Kushal Patel\Desktop\Courses\Computer Vision\Homework 2\FinalResults\Blurred.png", GaussianBlurred)

x_coord = loc[10, 5][0]
y_coord = loc[10, 5][1]
cv2.rectangle(image_cpy, (x_coord, y_coord),  (x_coord+ template.shape[1],  y_coord+ template.shape[0]), (0, 255, 0))
cv2.imwrite(r"C:\Users\Kushal Patel\Desktop\Courses\Computer Vision\Homework 2\FinalResults\Match.png", image_cpy)
res = np.uint8(save_res * 255)
cv2.imwrite(r"C:\Users\Kushal Patel\Desktop\Courses\Computer Vision\Homework 2\FinalResults\Res.png", res)
