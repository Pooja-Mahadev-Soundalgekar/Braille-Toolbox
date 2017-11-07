import cv
import cv2
import time
from PIL import Image
import numpy as np
import csv
import logistic
import mouthdetection as m
import math
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
    
    d_red = cv2.cv.RGB(150, 55, 65)
    l_red = cv2.cv.RGB(250, 200, 200)

    orig = cv2.imread("braillerstuv.jpg")
    orig = cv2.pyrUp(orig)
    img = orig.copy()
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector = cv2.FeatureDetector_create('MSER')
    fs = detector.detect(img2)
    fs.sort(key = lambda x: -x.size)
    def supress(x):
            for f in fs:
                distx = f.pt[0] - x.pt[0]
                disty = f.pt[1] - x.pt[1]
                dist = math.sqrt(distx*distx + disty*disty)
                if (f.size > x.size) and (dist<f.size/2):
                        return True

    sfs = [x for x in fs if not supress(x)]
    for f in sfs:
            cv2.circle(img, (int(f.pt[0]), int(f.pt[1])), int(f.size/2), d_red, 2, cv2.CV_AA)
            cv2.circle(img, (int(f.pt[0]), int(f.pt[1])), int(f.size/2), l_red, 1, cv2.CV_AA)
            x=int(f.pt[0])
            y=int(f.pt[1])
            r=int(f.size/2)
            print x,y,r
            img = cv.LoadImage("braille2.png")
            if(r<50):
                cropping=img[y-r: y + r+4, x-r: x + r+4]
                cv.SaveImage("crop1.jpg", cropping)
                result = lr.predict(vectorize('crop1.jpg'))
                if result == 1:
                    print "solid"
                else:
                    print "hollow"
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                print x,y,r
    cv2.imshow("output", np.hstack([image, output]))
    cv2.waitKey(0)