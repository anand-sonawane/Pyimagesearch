from transform import four_point_transform
import imutils
import numpy as np
import argparse
import cv2

#Code and tutorial reference :

#https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/?__s=q5pmdzt3javkrvysxqps

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

image_original = cv2.imread(args["image"])
image = image_original.copy()
#The steps we will go through are as follows:
#1.Edge detection
#2.Finding contours
#3.Apply prespective transform and threshold


#1.Edge detection
grayscale = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#grayscale = cv2.GaussianBlur(grayscale,(5,5),0)#Bluring is not required because canny applies blur on itself
edged = cv2.Canny(grayscale,100,250)

#2.Finding contours
#There are three arguments in cv2.findContours() function,
#first one is source image, second is contour retrieval mode, third is contour approximation method
im2, contours, hierarchy  = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#print(contours)
#print(hierarchy)
#The third argument saves the contour points for us

#Now we know that the bills will be rectangular in shape and therefore we found rectangular contours
#we will know find the biggest from them all.
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

# loop over the contours

for c in contours:
	# approximate the contour
	peri = cv2.arcLength(c, True) #arcLength gets the perimeter of the total contour
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)#approxPolyDP is used to get an approximate shape of the contour

	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
#Now we will draw the contour
image_draw = cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
#parameters : image, contour, drawing all contours then -1,color and then thickness

#3.Apply prespective transform and threshold
#Now we will apply four point transform to get prespective transform

warped = four_point_transform(image, screenCnt.reshape(4, 2))
#parameters image, contour which is reshaped to (4,2)

#then these are additional changes to make it look like a scanned copy

warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,2)

cv2.imshow("Original Image",image_original)
cv2.imshow("Image Edge Detected",edged)
cv2.imshow("Image with contour",image_draw)
cv2.imshow("Scanned Image",warped)
cv2.imwrite("Images/Scanned_Image.jpg",warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
