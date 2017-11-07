import cv
import numpy as np

# grab image
orig = cv.LoadImage('case1.jpg')

# create tmp images
grey_scale = cv.CreateImage(cv.GetSize(orig), 8, 1)
processed = cv.CreateImage(cv.GetSize(orig), 8, 1)

cv.Smooth(orig, orig, cv.CV_GAUSSIAN, 3, 3)

cv.CvtColor(orig, grey_scale, cv.CV_RGB2GRAY)

# do some processing on the grey scale image
cv.Erode(grey_scale, processed, None, 10)
cv.Dilate(processed, processed, None, 10)
cv.Canny(processed, processed, 5, 70, 3)
cv.Smooth(processed, processed, cv.CV_GAUSSIAN, 15, 15)

#storage = cv.CreateMat(orig.width, 1, cv.CV_32FC3)
storage = cv.CreateMemStorage(0)

contours = cv.FindContours(processed, storage, cv.CV_RETR_EXTERNAL)
# N.B. 'processed' image is modified by this!

#contours = cv.ApproxPoly (contours, storage, cv.CV_POLY_APPROX_DP, 3, 1) 
# If you wanted to reduce the number of points...

cv.DrawContours (orig, contours, cv.RGB(0,255,0), cv.RGB(255,0,0), 2, 3, cv.CV_AA, (0, 0)) 

def contour_iterator(contour):
  while contour:
    yield contour
    contour = contour.h_next()

for c in contour_iterator(contours):
  # Number of points must be more than or equal to 6 for cv.FitEllipse2
  if len(c) >= 6:
    # Copy the contour into an array of (x,y)s
    PointArray2D32f = cv.CreateMat(1, len(c), cv.CV_32FC2)

    for (i, (x, y)) in enumerate(c):
      PointArray2D32f[0, i] = (x, y)

    # Fits ellipse to current contour.
    (center, size, angle) = cv.FitEllipse2(PointArray2D32f)

    # Convert ellipse data from float to integer representation.
    center = (cv.Round(center[0]), cv.Round(center[1]))
    size = (cv.Round(size[0] * 0.5), cv.Round(size[1] * 0.5))

    # Draw ellipse
    cv.Ellipse(orig, center, size, angle, 0, 360, cv.RGB(255,0,0), 2,cv.CV_AA, 0)

# show images
cv.ShowImage("image - press 'q' to quit", orig)
#cv.ShowImage("post-process", processed)
cv.WaitKey(-1)