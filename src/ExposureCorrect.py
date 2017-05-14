import numpy as np 
from scipy.optimize import minimize

# Pre-requisites
# calculate region of overlap for all images
# Calculate average intensities in region of overlap for all images

# Jump exposure correction:
# Setup optimization problem in scipy
# We get a vector of gains - one gain value per camera.
# Final exposure tone mapping:
# For every column in the stitched panaroma, 
# Find the two gain values that fall on either side of the column
# Take weighted average of the two gain values and apply gain to this column (divide pixel values by new gain)

class ImageIntensityPair:
	def __init__(self):
		self.leftOverlapMean = 0
		self.rightOverlapMean = 0

	def __init__(self, lval, rval):
		self.leftOverlapMean = lval
		self.rightOverlapMean = rval

	def setLeftOverlap(self, val):
		self.leftOverlapMean = val

	def setRightOverlap(self, val):
		self.rightOverlapMean = val

	def setOverlapValues(self, vals):
		if len(vals) != 2:
			raise RuntimeError("Only 2D overlaps supported.")
		self.leftOverlapMean = vals[0]
		self.rightOverlapMean = vals[1]

	def getOverlapValues(self):
		return [self.leftOverlapMean, self.rightOverlapMean]

	def getRightOverlap(self):
		return self.rightOverlapMean

	def getLeftOverlap(self):
		return self.leftOverlapMean

# End ImageIntensityPair


# Loss function that is optimized during exposure correction.
def lossFunction(x, left_mean_intensities, right_mean_intensities):
	count = len(right_mean_intensities)
	cost = 0
	for i in range(count-1):
		cost = cost + (x[i]*right_mean_intensities[i] - x[i+1]*left_mean_intensities[i+1])**2
	# Regularization term
	cost = cost + 0.0001 *sum((1-x)**2)
	return cost


class OptimizeExposure:
	def __init__(self):
		self.left_mean_intensities = []
		self.right_mean_intensities = []
		self.image_count = 0

	def addImage(self, left_overlap_mean, right_overlap_mean):
		self.left_mean_intensities.append(left_overlap_mean)
		self.right_mean_intensities.append(right_overlap_mean)
		self.image_count = self.image_count+1

	def addImageList(self, left_om_vals, right_om_vals):
		if len(left_om_vals) != len(right_om_vals):
			raise RuntimeError("Length of both left and right mean intensities must be the same")
		self.left_mean_intensities.append(left_om_vals[:])
		self.right_mean_intensities.append(right_om_vals[:])
		self.image_count = self.image_count+len(left_om_vals)

	def addImageIntensityPairs(self, intensityPairs):
		for row in intensityPairs:
			self.left_mean_intensities.append(row[0])
			self.right_mean_intensities.append(row[1])
			self.image_count = self.image_count+1

	def getMeanIntensities(self):
		return [self.left_mean_intensities, self.right_mean_intensities]

	def calculateGains(self):
		initial_guess = np.ones((self.image_count))
		self.op_gains = minimize(lossFunction, initial_guess, args=(self.left_mean_intensities, self.right_mean_intensities, ), 
			method='nelder-mead', options={'xtol': 1e-8, 'disp': False})
		return self.op_gains.x


# end class exposure correction