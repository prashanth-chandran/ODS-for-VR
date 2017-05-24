import numpy as np
import imutils
import cv2


class Stitcher:
	def __init__(self):
		# Do nothing
		self.name = 'Stitcher'


	def stitch(self, 
				image1, 
				image2, 
				ratio=0.7, 
				reprojThresh=5.0, 
				showMatches = False):

		# detect key points and descriptors from the two images
		# that we need to stitch
		(kp1, feat1) = self.detectAndDescribe(image1)
		(kp2, feat2) = self.detectAndDescribe(image2)

		# Match features between the two images
		M = self.matchKeyPoints(kp1, kp2, feat1, feat2, ratio, reprojThresh)

		# If no matches were found, then we cannot create a panaroma
		if M is None:
			print('No matches found')
			return None

		# get the homography from the feature matching and use it to perspective warp the image
		(matches, H, status) = M
		result = np.zeros((image2.shape[1]+image1.shape[1], image2.shape[0]))
		size_res = (result.shape)
		temp = cv2.warpPerspective(image1, H, size_res)
		print(H)
		print(H.shape)
		result = temp
		result[0:image2.shape[0], 0:image2.shape[1]] = image2

		if showMatches:
			vis = self.drawMatches(image1, image2, kp1, kp2, matches, status)
			return (result, vis)

		return result

	def getKeyPointMatches(self, image1, image2, ratio=0.7, reprojThresh=5.0):
		(kp1, feat1) = self.detectAndDescribe(image1)
		(kp2, feat2) = self.detectAndDescribe(image2)

		M = self.matchKeyPoints(kp1, kp2, feat1, feat2, ratio, reprojThresh)

		return M

	def getKeyPoints(self, im1):
		(kp1, feat1) = self.detectAndDescribe(im1)
		return np.asarray(kp1)

	def detectAndDescribe(self, image):
		if len(image.shape) ==3:
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# openCV v3.x compatible code
		descriptor = cv2.xfeatures2d.SURF_create()
		(kps, features) = descriptor.detectAndCompute(image, None)

		kps = np.float32([kp.pt for kp in kps])

		return (kps, features)

	def matchKeyPoints(self, kp1, kp2, feat1, feat2, ratio, rThresh):
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(feat1, feat2, 2)
		matches = []

		for m in rawMatches:
			# Lowe's ratio test to detect false positives
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# print(len(matches), len(kp1), len(kp2))
		if len(matches) > 4:
			pts1 = np.float32([kp1[i] for (_, i) in matches])
			pts2 = np.float32([kp2[i] for (i, _) in matches])

			# compute homography
			(H, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC, rThresh)

			return (matches, H, status)

		return None


	def drawMatches(self, image1, image2, kp1, kp2, matches, status):
		(h1, w1) = image1.shape[:2]
		(h2, w2) = image2.shape[:2]
		vis = np.zeros((max(h1, h2), w1+w2, 3), dtype='uint8')
		vis[0:h1, 0:w1] = image1
		vis[0:h2, w1:] = image2

		for ((ti, qi), s) in zip(matches, status):
			if s==1:
				pt1 = (int(kp1[qi][0]), int(kp1[qi][1]))
				pt2 = (int(kp2[ti][0])+w1, int(kp2[ti][1]))
				cv2.line(vis, pt1, pt2, (255, 0, 0), 1)

		return vis

# End of class Stitcher


class ImageOverlapProcessor:
	def __init__(self, image_dim, direction='left'):
		self.img_dim = np.asarray(image_dim)
		self.x_dim = self.img_dim[1]
		self.y_dim = self.img_dim[0]
		self.direction = direction
		# Initialize to no overlap
		self.start = self.x_dim
		self.end = self.x_dim

	def getRegionOfOverlap(self):
		return np.asarray([self.start, self.end])

	def calculateRegionOfOverlap(self, keyPoints1, matches, status):
		kp1 = np.asarray(keyPoints1)
		kp1 = self.__filterKeyPoints(kp1, matches, status)
		num_keypoints1 = kp1.shape[0]
		if (kp1.shape[1] != 2):
			raise ValueError('Only 2D matches supported.')

		if self.direction is 'left':
			self.start = kp1[:, 0].min()
			self.end = kp1[:, 0].max()
		elif self.direction is 'right':
			self.end = kp1[:, 0].max()
			self.start = kp1[:, 0].min()


	def setOverlap(self, ostart, oend):
		self.start = ostart
		self.end = oend


	def __filterKeyPoints(self, keypoints, matches, status):
		k_filtered = []
		for ((ti, qi), s) in zip(matches, status):
			if s==1:
				if self.direction == 'left':
					pt = (int(keypoints[ti][0]), int(keypoints[ti][1]))
				elif self.direction == 'right':
					pt = (int(keypoints[qi][0]), int(keypoints[qi][1]))
				else:
					raise RuntimeError('Unknown direction in calculating image overlap')
				k_filtered.append(pt)

		return np.asarray(k_filtered)

	def getAverageOverlapIntensity(self, image):
		image_dim = image.shape
		if (self.start > image_dim[1] or self.end > image_dim[1]):
			raise RuntimeError("ImageOverlapProcessor's dimensions are bigger than that of the input image.")
		over_lap_im = image[:, self.start:self.end, :]
		mean_intensities = np.mean(over_lap_im, axis=-1)
		mean_intensity = np.mean(mean_intensities)
		return mean_intensity

# End of class ImageOverlap
