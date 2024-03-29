import numpy as np 
import cv2
import imutils
import os 
import os.path as op
import yaml
from Stitcher import *
from ExposureCorrect import *

class SJPImage:
	"""
	OpenCV image wrapper that's a little relevant to our project.
	"""
	def __init__(self, image=None, file_name=None, resize=True):
		self.file_name = file_name
		if (image is None) and (file_name is not None):
			self.readImage(file_name, resize)
		else:
			if image is None:
				raise RuntimeError("Both file name and image cannot be None")
			self.cv_image = image
			if resize:
				self.cv_image = imutils.resize(self.cv_image, width=400)
			self.rows = self.cv_image.shape[0]
			self.cols = self.cv_image.shape[1]
			self.channels = self.cv_image.shape[2]
			self.image_dim = [self.rows, self.cols, self.channels]

		self.cameraID = 'unknown'
		self.frameID = 0
		self.left_overlap_proc = ImageOverlapProcessor(self.image_dim, direction='left')
		self.right_overlap_proc = ImageOverlapProcessor(self.image_dim, direction='right')
		self.gain_left = 1
		self.gain_self = 1
		self.gain_right = 1
		self.left_image = None
		self.right_image = None
		self.H_left = None
		self.H_right = None
		self.Matches_left = None
		self.Matches_right = None

	def initializeImage(self, left_image=None, right_image=None):
		stitcher = Stitcher()
		self.setKeypoints(stitcher.getKeyPoints(self.cv_image))
		if left_image is not None:
			self.left_image = left_image
		if right_image is not None:
			self.right_image = right_image
		if right_image is not None and left_image is not None:
			self.updateOverlappingRegions(self.left_image, self.right_image)

	def readImage(self, file_name, resize=True):
		self.cv_image = cv2.imread(file_name)
		if self.cv_image is None:
			raise RuntimeError('Image does not exist.')
		if resize:
			self.cv_image = imutils.resize(self.cv_image, width=400)
		self.rows = self.cv_image.shape[0]
		self.cols = self.cv_image.shape[1]
		self.channels = self.cv_image.shape[2]
		self.image_dim	= [self.rows, self.cols, self.channels]

	def updateOverlappingRegions(self, leftImage=None, rightImage=None):
		if (leftImage is None and rightImage is None):
			raise RuntimeError('Both left and right images cannot be empty')
		if (self.keypoints is None):
			raise RuntimeError('Keypoints must be set for this image before calling this function')

		stitcher = Stitcher()
		valLeft = 0
		valRight = 0
		if (leftImage is not None):
			(matches, H, status) = stitcher.getKeyPointMatches(self.getImage(), leftImage.getImage())
			# Calculate region of overlap after keypoint matching
			ov_proc = ImageOverlapProcessor(self.getImageDim(), direction='right')
			ov_proc.calculateRegionOfOverlap(self.getKeypoints(), matches, status)
			valLeft = ov_proc.getAverageOverlapIntensity(self.getImage())
			self.H_left = H
			self.Matches_right = matches
		
		if (rightImage is not None):
			(matches, H, status) = stitcher.getKeyPointMatches(rightImage.getImage(), self.getImage())
			# Calculate region of overlap after keypoint matching
			ov_proc = ImageOverlapProcessor(self.getImageDim(), direction='left')
			ov_proc.calculateRegionOfOverlap(self.getKeypoints(), matches, status)
			valRight = ov_proc.getAverageOverlapIntensity(self.getImage())
			self.H_right = H
			self.Matches_right = matches

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

	def setNeighbours(self, im_left, im_right):
		self.left_image = im_left
		self.right_image = im_right

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

	def getHomographyLeft(self):
		return self.H_left

	def getHomographyRight(self):
		return self.H_right

	def getColumn(self, column_index):
		if column_index >= 0 and column_index < self.cols:
			return self.cv_image[:, column_index, :]
		else:
			raise RuntimeError('Index exceeds image dimensions')

	def imshow(self, meta_data):
		if self.file_name is None:
			self.file_name = 'Unknown'
		title_ = op.basename(self.file_name) + ' cam: '  + str(self.getCameraID()) + ' f: ' + str(self.getFrameID())
		if meta_data is not None:
			title_ = title_ + meta_data
		cv2.imshow(title_, self.cv_image)

	def __str__(self):
		return self.file_name

# End class SJPImage


class SJPImageCollection:
	def __init__(self, sjp_image_list = None):
		self.sjp_image_list = []
		self.frameID = -1
		self.num_images = 0

	
	def addImageToCollection(self, sjp_image):
		self.sjp_image_list.append(sjp_image)
		self.num_images = self.num_images + 1


	def loadImagesFromYAML(self, file_name, cam_name):
		with open(file_name, 'r') as stream:
			try:
				calib_data = yaml.load(stream)
			except Exception as e:
				raise RuntimeError('Something bad happened when loading the .yaml file')

		image_names = calib_data[cam_name]['images']
		for image in image_names:
			print('reading image: ', image)
			im_new = SJPImage(file_name=image, resize=False)
			im_new.initializeImage()
			im_new.setFrameID(cam_name)
			self.addImageToCollection(im_new)


	def setFrameID(self, frameID):
		self.frameID = frameID

	def getNumberOfImages(self):
		return self.num_images

	def __getitem__(self, key):
		return self.sjp_image_list[key]

	def __len__(self):
		return self.num_images


# End class SJPImageCollection

