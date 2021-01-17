from __future__ import print_function
import sys
import numpy as np
import cv2
from random import randint

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv2.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]: 
    tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)
    
  return tracker

# Set video to load
videoPath = "test.mp4"

# Create a video capture object to read videos
cap = cv2.VideoCapture(videoPath)

## Select boxes
bboxes = []
colors = [] 

print('Selected bounding boxes {}'.format(bboxes))

canvas = np.zeros((768,1024,3), np.uint8)
car = cv2.imread('car.png')

# cv2.imshow('car', canvas)
# cv2.waitKey()

# receive raw_loc from tracker algorithm running in lower camera
# map bbox in lower camera into the roi in higher camera
raw_loc = (0,0,car.shape[1],car.shape[0])

def map_bbox(xyxy):
	def get_transform_between_cameras():
		M = cv2.getPerspectiveTransform(camera1_pixels,camera2_pixels)
		return M
	
	camera1_pixels = np.float32([[0,0],[800,0],[0,600],[800,600]])
	camera2_pixels = np.float32([[200,100],[420,110],[230,150],[450,500]])

	M = get_transform_between_cameras()
	def roi_map(point, T):
		pt = np.float32([[point[0]],[point[1]],[1]])
		ret = np.dot(T, pt)
		ret = ret/ret[2]
		return ret[:,0]
	bb = xyxy 
	ret = roi_map([bb[0],bb[1]],M)
	min_x,min_y = ret[:2]
	max_x,max_y = ret[:2]
	
	ret = roi_map([bb[2],bb[1]],M)
	min_x = min(min_x,ret[0]);min_y = min(min_y,ret[1])
	max_x = min(max_x,ret[0]);max_y = max(max_y,ret[1])
	ret = roi_map([bb[0],bb[3]],M)
	min_x = min(min_x,ret[0]);min_y = min(min_y,ret[1])
	max_x = min(max_x,ret[0]);max_y = max(max_y,ret[1])
	ret = roi_map([bb[2],bb[3]],M)
	min_x = min(min_x,ret[0]);min_y = min(min_y,ret[1])
	max_x = min(max_x,ret[0]);max_y = max(max_y,ret[1])
	return [min_x,min_y,max_x,max_y]

# loc = map_bbox(raw_loc)
loc = raw_loc 
bboxes.append(loc)
colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))

# Specify the tracker type
trackerType = "MIL"    

frame = canvas.copy()
frame[loc[1]:loc[1]+loc[3],loc[0]:loc[0]+loc[2]] = car
# cv2.imshow('car', car)
# cv2.imshow('MultiTracker', frame[loc[0]:loc[0]+loc[2],loc[1]:loc[1]+loc[3]])
# cv2.waitKey()
# Create MultiTracker object
trackers = []
# Initialize MultiTracker 
for bbox in bboxes:
	tracker = createTrackerByName(trackerType)
	tracker.init(frame, bbox)
	trackers.append(tracker) 

# Process video and track objects
while cap.isOpened():
	success, frame = cap.read()
	if not success:
		break

	# simulate next frame
	frame = canvas.copy()

	for id, bb in enumerate(bboxes):
		loc = bboxes[id]
		bboxes[id] = (loc[0]+5,loc[1],loc[2],loc[3])
		loc = bboxes[id]
		print(loc)
		frame[loc[1]:loc[1]+loc[3],loc[0]:loc[0]+loc[2]] = car
	
	# tracking
	for tracker in trackers:
		(success, box) = tracker.update(frame)
		if success:
			(x, y, w, h) = [int(v) for v in box]
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		else:
			# remove tracker for many failures
			# TODO
			pass
	# show frame
	cv2.imshow('MultiTracker', frame)
	

	# quit on ESC button
	if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
	  break