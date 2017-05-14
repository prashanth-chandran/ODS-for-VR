import numpy as np 
import cv2
import imutils
import os 
import os.path as op
from Stitcher import *
from ExposureCorrect import *

class SJPImage:
	"""
	OpenCV image wrapper that's a little relevant to our project.
	"""
	def __init__(self, file_name, resize=True):
		self.file_name = file_name
		self.readImage(file_name, resize)
		self.cameraID = 0
		self.frameID = 0
		self.left_overlap_proc = ImageOverlapProcessor(self.image_dim, direction='left')
		self.right_overlap_proc = ImageOverlapProcessor(self.image_dim, direction='right')
		self.gain_left = 1
		self.gain_self = 1
		self.gain_right = 1

	def readImage(self, file_name, resize=True):
		self.cv_image = cv2.imread(file_name)
		if resize:
			self.cv_image = imutils.resize(self.cv_image, width=400)
		self.rows = self.cv_image.shape[0]
		self.cols = self.cv_image.shape[1]
		self.channels = self.cv_image.shape[2]
		self.image_dim	= [self.rows, self.cols, self.channels]

	def updateOverlappingRegions(self, leftImage=None, rightImage=None):
		if (leftImage is None and rightImage is None):
			raise RuntimeError('Both left and right images cannot be empty')

		stitcher = Stitcher()
		valLeft = 0
		valRight = 0
		if (leftImage is not None):
			(matches, H, status) = stitcher.getKeyPointMatches(self.getImage(), leftImage.getImage())
			# Calculate region of overlap after keypoint matching
			ov_proc = ImageOverlapProcessor(self.getImageDim(), direction='right')
			ov_proc.calculateRegionOfOverlap(self.getKeypoints(), matches, status)
			valLeft = ov_proc.getAverageOverlapIntensity(self.getImage())
		
		if (rightImage is not None):
			(matches, H, status) = stitcher.getKeyPointMatches(rightImage.getImage(), self.getImage())
			# Calculate region of overlap after keypoint matching
			ov_proc = ImageOverlapProcessor(self.getImageDim(), direction='left')
			ov_proc.calculateRegionOfOverlap(self.getKeypoints(), matches, status)
			valRight = ov_proc.getAverageOverlapIntensity(self.getImage())

		self.setImageIntensityOverlap(ImageIntensityPair(valLeft, valRight))

	def exposureCorrectJumpStyle(self):
		# Go over every column in the image. 
		# For every column index, calculate the weighted gain for this column
		for i in range(self.cols):
			gain = self.__calculateWeightedGain(i)
			self.cv_image[:, i, :] = self.cv_image[:, i, :] * gain

	def __calculateWeightedGain(self, col_index):
		mid_point = float(self.cols/2)
		gain = 1
		gain_neighbour = 1
		w = 1.0
		w_neighbour = 1.0
		if (col_index < mid_point):
			w = float(col_index/mid_point) 
			w_neighbour = (1-w)
			gain_neighbour = self.gain_left
		else:
			w = float((self.cols - col_index)/float(self.cols))+0.5
			w_neighbour = (1-w)
			gain_neighbour = self.gain_right
		
		if (w+w_neighbour != 1):
			raise RuntimeError('Weight assignment is bad !', w, w_neighbour)

		gain = (w*self.gain_self) + (w_neighbour * gain_neighbour)
		return gain


	def writeImage(self, file_name):
		pass

	def setFrameID(self, fid):
		self.frameID = fid

	def setCameraID(self, cid):
		self.cameraID = cid

	def setKeypoints(self, keypoints):
		self.keypoints = keypoints

	def setImageIntensityOverlap(self, image_intensity_pair):
		self.overlap_vals = image_intensity_pair

	def setNeighbourGains(self, gainLeft, gainRight):
		self.gain_left = gainLeft
		self.gain_right = gainRight

	def setGain(self, gain):
		self.gain_self = gain

	def getImage(self):
		return self.cv_image

	def getCameraID(self):
		return self.cameraID

	def getFrameID(self):
		return self.frameID

	def getKeypoints(self):
		return self.keypoints

	def getImageDim(self):
		return self.image_dim

	def getImageOverlapMeans(self):
		return self.overlap_vals.getOverlapValues()

	def imshow(self):
		title_ = op.basename(self.file_name) + ' cam: '  + str(self.getCameraID()) + ' f: ' + str(self.getFrameID())
		cv2.imshow(title_, self.cv_image)

# End class SJPImage
