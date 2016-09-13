#################################################################################
#
#	Ball tracking with Kinect - X,Y,Z
#
#	Description: Program to detect the position of a ball in space using the 
#				 Kinect sensor using the OpenCv(Computer Vision Library) and
#				 freenect(Open Source Kinect library).
#
# 	Date: 09/2016
#   Authors: Gustavo Carlos, Paulo Custodio
#		
#   LRVA - Laboratorio de Robotica e Veiculos Autonomos
################################################################################


#Values for Red ball
hmin = 163
hmax = 180
vmin = 176
vmax = 200
smin = 139
smax = 224

#Values Helosman Experiment
# hmin 144
# hmax 255
# smin 196
# smax 255
# vmin 131
# vmax 191

#impor required libraries
import numpy as np # numpy for array manipulation
import cv2	#OpenCv
import cv2.cv as cv #Open Cv
import math

import freenect # Kinect free library

# function to write a string in a frame
def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.CV_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)


#function to get RGB image from kinect
def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array
    
#function to get depth image from kinect
def get_depth():
    array,_ = freenect.sync_get_depth()
    array = array.astype(np.uint8)
    return array

#function to get distances in cm, equation from OpenKinect    
def get_distance_St(depthRaw):
	depthM = 0.1236 * math.tan(depthRaw / 2842.5 + 1.1863)
	return depthM * 100

#function to cal real X from pixel and Z distance in centimeters
def calcXCM(pixelX, distZ):
	xCm = (pixelX - 320) * (distZ - 10) * 0.0021
	return xCm
	
#function to cal real Y from pixel and Z distance in centimeters	
def calcYCM(pixelY, distZ):
	yCm = (pixelY - 240) * (distZ - 10) * 0.0021
	return yCm

# function to be passed as parameter in trackBars initialization
def nothing(x):
    pass

#vectors to use in openCv functions
kernel = np.ones((5,5),np.uint8)
kernel2 = np.ones((20,20),np.uint8)

#Windows to show images
cv2.namedWindow("Original")
cv2.namedWindow('HueAdj')
cv2.namedWindow("SatAdj")
cv2.namedWindow("ValAdj")
cv2.namedWindow("Closing")
cv2.namedWindow("Depth")


#create track bars to adjust H, S and V of images
cv2.createTrackbar('hmin', 'HueAdj',0,180,nothing)
cv2.createTrackbar('hmax', 'HueAdj',180,180,nothing)

cv2.createTrackbar('smin', 'SatAdj',0,255,nothing)
cv2.createTrackbar('smax', 'SatAdj',255,255,nothing)

cv2.createTrackbar('vmin', 'ValAdj',0,255,nothing)
cv2.createTrackbar('vmax', 'ValAdj',255,255,nothing)

cv2.setTrackbarPos('hmin','HueAdj', hmin)
cv2.setTrackbarPos('hmax', 'HueAdj', hmax)
cv2.setTrackbarPos('smin', 'SatAdj', smin)
cv2.setTrackbarPos('smax', 'SatAdj', smax)
cv2.setTrackbarPos('vmin', 'ValAdj', vmin)
cv2.setTrackbarPos('vmax', 'ValAdj', vmax)

logData = open('MAX.txt', 'w')


countData = 0

#infinity loop
while countData < 100:

	frame = get_video() #get RGB image from kinect
	depth = get_depth() #get Depth image normalized from kinect, just to show
	
	depthOriginal,_ = freenect.sync_get_depth() # get 11 bit depth value from kinect
	
	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) #convert RGB image to HSV domain
	
	#get hue, sat and val separately from HSV image
	hue,sat,val = cv2.split(hsv)
	
	# get values min and max from trackBars 
	hmn = cv2.getTrackbarPos('hmin','HueAdj') #144#
	hmx = cv2.getTrackbarPos('hmax','HueAdj') #180#
	smn = cv2.getTrackbarPos('smin','SatAdj') #196#
	smx = cv2.getTrackbarPos('smax','SatAdj') #255#
	vmn = cv2.getTrackbarPos('vmin','ValAdj') #131#
	vmx = cv2.getTrackbarPos('vmax','ValAdj') #191#
	
	# Apply thresholding
	hthresh = cv2.inRange(np.array(hue),np.array(hmn),np.array(hmx))
	sthresh = cv2.inRange(np.array(sat),np.array(smn),np.array(smx))
	vthresh = cv2.inRange(np.array(val),np.array(vmn),np.array(vmx))
	
	# AND h s and v
	tracking = cv2.bitwise_and(hthresh,cv2.bitwise_and(sthresh,vthresh))
	
	# Some morpholigical filtering
	dilation = cv2.dilate(tracking,kernel,iterations = 1)
	#erode = cv2.erode(tracking, kernel, iterations = 1)
	closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
	closing = cv2.GaussianBlur(closing,(5,5),0)
	
	
	# Detect circles using HoughCircles
	circles = cv2.HoughCircles(closing,cv.CV_HOUGH_GRADIENT,2,120,param1=120,param2=50,minRadius=10,maxRadius=0)
	
	if circles is not None:
		for i in circles[0,:]:
			#draw_str(frame,(int(round(i[1]+i[2])), int(round(i[0]+i[2]))), 'x: ' + str(i[0]) + ' y: ' + str(i[1]))
			

			#get distance for each pixel
			
			#print distance for a center pixel of sphere
			if(i[0] < 600):
				zCM = get_distance_St(depthOriginal[int(round(i[1]))][int(round(i[0])) + 25])
				draw_str(frame, (int(round(i[1]+i[2])), int(round(i[0]+i[2]))), '%.2f' % zCM)
				xCM = calcXCM(i[0],zCM)
				yCM = calcYCM(i[1],zCM)
				print 'x: %f y: %f z: %f' % (xCM, yCM, zCM)
				#print zCM
				#logData.write('%.4f\n' % zCM)
				#countData = countData + 1
			
			
			# draw a circle around the object in the original image
			cv2.circle(frame,(int(round(i[0])),int(round(i[1]))),int(round(i[2])),(0,255,0),5)
			cv2.circle(frame,(int(round(i[0])),int(round(i[1]))),2,(0,0,255),10)
			# draw a circle around the object in the depth image
			# todo: verify the diference between 2 frames
			cv2.circle(depth,(int(round(i[0])),int(round(i[1]))),int(round(i[2])),(0,255,0),5)
			cv2.circle(depth,(int(round(i[0])),int(round(i[1]))),2,(0,0,255),10)

    #Show the result  frames
	cv2.imshow('HueAdj',hthresh)
	cv2.imshow('SatAdj',sthresh)
	cv2.imshow('ValAdj',vthresh)
	cv2.imshow('Closing',closing)
	cv2.imshow('Original',frame)
	cv2.imshow('Depth', depth)
		
	#wait some time
	key_pressed = cv2.waitKey(1)
	
logData.close()

