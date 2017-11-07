import cv
import cv2
import time
from PIL import Image
import numpy as np
import csv
import logistic
import mouthdetection as m
from math import sin, cos, radians
import argparse
from operator import itemgetter
WIDTH, HEIGHT = 28, 10 
dim = WIDTH * HEIGHT 

def vectorize(filename):
    size = WIDTH, HEIGHT 
    im = Image.open(filename) 
    resized_im = im.resize(size, Image.ANTIALIAS) 
    im_grey = resized_im.convert('L') 
    im_array = np.array(im_grey) 
    oned_array = im_array.reshape(1, size[0] * size[1])
    return oned_array
if __name__ == '__main__':
    smilefiles = []
    with open('smiles.csv', 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            smilefiles += rec

    neutralfiles = []
    with open('neutral.csv', 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            neutralfiles += rec       
    phi = np.zeros((len(smilefiles) + len(neutralfiles), dim))
    labels = []
    PATH = "../data/smile/"
    for idx, filename in enumerate(smilefiles):
        phi[idx] = vectorize(PATH + filename)
        labels.append(1)  
    PATH = "../data/neutral/"
    offset = idx + 1
    for idx, filename in enumerate(neutralfiles):
        phi[idx + offset] = vectorize(PATH + filename)
        labels.append(0)

    lr = logistic.Logistic(dim)
    lr.train(phi, labels)
    image = cv2.imread('braillerstuv.jpg')
    output = cv2.imread('braillerstuv.jpg')
    image = cv2.pyrUp(image)
    output = cv2.pyrUp(output)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 100)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        circles=sorted(circles, key=itemgetter(0,1))
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle

            cv2.circle(output, (x, y), r+4, (0, 255, 50), 2)
            img = cv.LoadImage("imga.jpg")
            cropping=img[y-r-4: y + r+4, x-r-4: x + r+4]
            cv.SaveImage("crop1.jpg", cropping)
            result = lr.predict(vectorize('crop1.jpg'))
            if result == 1:
            	print "solid"
            else:
            	print "hollow"
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            print x,y,r
        # show the output image
        cv2.imshow("output", np.hstack([image, output]))
        cv2.waitKey(0)