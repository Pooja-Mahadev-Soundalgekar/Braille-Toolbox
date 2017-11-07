import cv2
import time
from PIL import Image
import numpy as np
import csv
import logistic
import math
import time
import argparse
import sys
from operator import itemgetter, attrgetter, methodcaller
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
    timer1=time.time()
    lr = logistic.Logistic(dim)
    lr.train(phi, labels)
    timer2=time.time()
    print "The time taken for logistic regression is ",timer2-timer1
    timer1=time.time()
    d_red = cv2.cv.RGB(150, 55, 65)
    l_red = cv2.cv.RGB(250, 200, 200)

    orig = cv2.imread(sys.argv[2])
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
   
    CircleDetail=[]
    count=0
    for f in sfs:
            cv2.circle(img, (int(f.pt[0]), int(f.pt[1])), int(f.size/2), d_red, 2, cv2.CV_AA)
            cv2.circle(img, (int(f.pt[0]), int(f.pt[1])), int(f.size/2), l_red, 1, cv2.CV_AA)
            x=int(f.pt[0])
            y=int(f.pt[1])
            r=int(f.size/2)
            xyrlist=[]
            xyrlist.insert(0,x)
            xyrlist.insert(1,y)
            xyrlist.insert(2,r)
            CircleDetail.insert(count, xyrlist)
            count=count+1
    timer2=time.time()
    print "The time taken to detect the braille holes ",timer2-timer1
    CircleDetail=sorted(CircleDetail, key=itemgetter(0,1))
    numberofcircle=sys.argv[1]
    print "The accuracy is:", 1.0-abs((int(numberofcircle)*1.0-len(CircleDetail)*1.0))/(len(CircleDetail)*1.0)
    i=0
    j=0
    matr=[0,0,0]
    maty=[0,0,0]
    matx=[0,0,0]
    logic=[0,0,0]
    for x,y,r in CircleDetail:
            if(r<100):

                img1 = cv2.imread(sys.argv[2])
                cropping=img1[y-r-4: y + r+4, x-r-4: x + r+4]
                cv2.imwrite("crop1.jpg", cropping)
                #cropping=orig[y-r-4: y + r+4, x-r-4: x + r+4]
                #cropping=cv.fromarray(cropping)
                #cv.SaveImage("crop1.png", cropping)
                result = lr.predict(vectorize('crop1.jpg'))

                if(i==3):
                	i=0
                	j=j+1
                	if(j==2):
                		j=0
                matx[i]=x
                maty[i]=y
                matr[i]=r
                if result == 1:
                	logic[i]=1
                	#print "solid"
                else:
                	logic[i]=0
                	#print "hollow"

        		
            	if(i==2):
            		if(maty[0]>maty[1] and maty[0]>maty[2]):
            			if(maty[1]>maty[2]):
            				temp1=matx[2]
            				temp2=maty[2]
            				temp3=matr[2]
            				temp4=logic[2]
            				matx[2]=matx[0]
            				maty[2]=maty[0]
            				matr[2]=matr[0]
            				logic[2]=logic[0]
            				matx[0]=temp1
            				maty[0]=temp2
            				matr[0]=temp3
            				logic[0]=temp4
            			else:
            				flag=2
            				temp1=matx[1]
            				temp2=maty[1]
            				temp3=matr[1]
            				temp4=logic[1]
            				matx[1]=matx[2]
            				maty[1]=maty[2]
            				matr[1]=matr[2]
            				logic[1]=logic[2]
            				matx[2]=matx[0]
            				maty[2]=maty[0]
            				matr[2]=matr[0]
            				logic[2]=logic[0]
            				matx[0]=temp1
            				maty[0]=temp2
            				matr[0]=temp3
            				logic[0]=temp4
            		else:
            			if(maty[1]>maty[2]):
            				if(maty[0]>maty[2]):
	            				flag=3
	            				temp1=matx[1]
	            				temp2=maty[1]
	            				temp3=matr[1]
	            				temp4=logic[1]
	            				matx[1]=matx[0]
	            				maty[1]=maty[0]
	            				matr[1]=matr[0]
	            				logic[1]=logic[0]
	            				matx[0]=matx[2]
	            				maty[0]=maty[2]
	            				matr[0]=matr[2]
	            				logic[0]=logic[2]
	            				matx[2]=temp1
	            				maty[2]=temp2
	            				matr[2]=temp3
	            				logic[2]=temp4
	            			else:
	            				flag=4
	            				temp1=matx[2]
	            				temp2=maty[2]
	            				temp3=matr[2]
	            				temp4=logic[2]
	            				matx[2]=matx[1]
	            				maty[2]=maty[1]
	            				matr[2]=matr[1]
	            				logic[2]=logic[1]
	            				matx[1]=temp1
	            				maty[1]=temp2
	            				matr[1]=temp3
	            				logic[1]=temp4
	            		else:
	            			if(maty[0]>maty[1]):
	            				flag=5
	            				temp1=matx[1]
	            				temp2=maty[1]
	            				temp3=matr[1]
	            				temp4=logic[1]
	            				matx[1]=matx[0]
	            				maty[1]=maty[0]
	            				matr[1]=matr[0]
	            				logic[1]=logic[0]
	            				matx[0]=temp1
	            				maty[0]=temp2
	            				matr[0]=temp3
	            				logic[0]=temp4
	            			else:
	            				flag=6

	            	if(j==0):
	            		a1=logic[0]
	            		a2=logic[1]
	            		a3=logic[2]
	            	#print a1,a2,a3,logic[0],logic[1],logic[2],matx[0],matx[1],matx[2],maty[0],maty[1],maty[2]
	            	if(j==1):
	            		if(a1==1 and a2==0 and a3==0 and logic[0]==1 and logic[1]==1 and logic[2]==0 ):
	            			print "4D"
	            		if(a1==1 and a2==0 and a3==0 and logic[0]==0 and logic[1]==1 and logic[2]==0 ):
	            			print "5E"
	            		if(a1==1 and a2==1 and a3==0 and logic[0]==1 and logic[1]==0 and logic[2]==0 ):
	            			print "6F"
	            		if(a1==1 and a2==1 and a3==0 and logic[0]==1 and logic[1]==1 and logic[2]==0 ):
	            			print "7G"
	            		if(a1==1 and a2==1 and a3==0 and logic[0]==0 and logic[1]==1 and logic[2]==0 ):
	            			print "8H"
	            		if(a1==0 and a2==1 and a3==0 and logic[0]==1 and logic[1]==0 and logic[2]==0 ):
	            			print "9I"
	            		if(a1==0 and a2==1 and a3==0 and logic[0]==1 and logic[1]==1 and logic[2]==0 ):
	            			print "0J"
	            		if(a1==1 and a2==0 and a3==0 and logic[0]==0 and logic[1]==0 and logic[2]==0 ):
	            			print "1A"
	            		if(a1==1 and a2==1 and a3==0 and logic[0]==0 and logic[1]==0 and logic[2]==0 ):
	            			print "2B"
	            		if(a1==1 and a2==0 and a3==0 and logic[0]==1 and logic[1]==0 and logic[2]==0 ):
	            			print "3C"
	            		if(a1==1 and a2==0 and a3==1 and logic[0]==0 and logic[1]==0 and logic[2]==0 ):
	            			print "K"
	            		if(a1==1 and a2==1 and a3==1 and logic[0]==0 and logic[1]==0 and logic[2]==0 ):
	            			print "L"
	            		if(a1==1 and a2==0 and a3==1 and logic[0]==1 and logic[1]==0 and logic[2]==0 ):
	            			print "M"
	            		if(a1==1 and a2==0 and a3==1 and logic[0]==1 and logic[1]==1 and logic[2]==0 ):
	            			print "N"
	            		if(a1==1 and a2==0 and a3==1 and logic[0]==0 and logic[1]==1 and logic[2]==0 ):
	            			print "O"
	            		if(a1==1 and a2==1 and a3==1 and logic[0]==1 and logic[1]==0 and logic[2]==0 ):
	            			print "P"
	            		if(a1==1 and a2==1 and a3==1 and logic[0]==1 and logic[1]==1 and logic[2]==0 ):
	            			print "Q"
	            		if(a1==1 and a2==1 and a3==1 and logic[0]==0 and logic[1]==1 and logic[2]==0 ):
	            			print "R"
	            		if(a1==0 and a2==1 and a3==1 and logic[0]==1 and logic[1]==0 and logic[2]==0 ):
	            			print "S"
	            		if(a1==0 and a2==1 and a3==1 and logic[0]==1 and logic[1]==1 and logic[2]==0 ):
	            			print "T"
	            		if(a1==1 and a2==0 and a3==1 and logic[0]==0 and logic[1]==0 and logic[2]==1 ):
	            			print "U"
	            		if(a1==1 and a2==1 and a3==1 and logic[0]==0 and logic[1]==0 and logic[2]==1 ):
	            			print "V"
	            		if(a1==0 and a2==1 and a3==0 and logic[0]==1 and logic[1]==1 and logic[2]==1 ):
	            			print "W"
	            		if(a1==1 and a2==0 and a3==1 and logic[0]==1 and logic[1]==0 and logic[2]==1 ):
	            			print "X"
	            		if(a1==1 and a2==0 and a3==1 and logic[0]==1 and logic[1]==1 and logic[2]==1 ):
	            			print "Y"
	            		if(a1==1 and a2==0 and a3==1 and logic[0]==0 and logic[1]==1 and logic[2]==1):
	            			print "Z"
	            		if(a1==0 and a2==0 and a3==1 and logic[0]==1 and logic[1]==1 and logic[2]==1 ):
	            			print "#"


            				


            	i=i+1 

                #cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                
    
    h, w = orig.shape[:2]
    vis = np.zeros((h, w*2+5), np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    vis[:h, :w] = orig
    vis[:h, w+5:w*2+5] = img

    cv2.imshow("image", vis)
    cv2.waitKey(0)