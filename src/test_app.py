from Stitcher import *
from SJPImage import *
from viewSynth import *
import argparse

def arg_setup():
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--first", required=True, help="path to the first image")
	ap.add_argument("-s", "--second", required=True, help="path to the second image")
	ap.add_argument("-r", "--result", required=False, help="path to save the panaroma")
	args = vars(ap.parse_args())
	return args


def test_overlap():
	args = arg_setup()
	# Image IO
	image1 = cv2.imread(args["first"])
	print(args["first"], args["second"])
	image2 = cv2.imread(args["second"])
	image1 = imutils.resize(image1, width=400)
	image2 = imutils.resize(image2, width=400)
	print(image1.shape, image2.shape)

	# Call our stitch function
	stitcher = Stitcher()
	(matches, homography, status) =stitcher.getKeyPointMatches(image2, image1)
	matches = np.asarray(matches)
	if (matches.shape[0] == 0):
		raise RuntimeError('No keypoint matches found !')

	# Find the region of overlap between for this image
	im_ov = ImageOverlapProcessor(image1.shape, direction='left')
	im_kp = stitcher.getKeyPoints(image1)
	im2_kp = stitcher.getKeyPoints(image2)
	im_ov.calculateRegionOfOverlap(im_kp, matches, status)
	print('Image 1:')
	print(im_ov.getRegionOfOverlap())
	print('average overlap intensity: ', im_ov.getAverageOverlapIntensity(image1))
	im_ov2 = ImageOverlapProcessor(image2.shape, direction='right')
	print('Image 2:')
	im_ov2.calculateRegionOfOverlap(im2_kp, matches, status)
	print(im_ov2.getRegionOfOverlap())
	print('average overlap intensity: ', im_ov2.getAverageOverlapIntensity(image2))


def test_SJPFramework():
	# Basic tests for the SJPImage class
	args = arg_setup()
	image1 = args["first"]
	image2 = args["second"]
	sim1 = SJPImage(image1)
	sim2 = SJPImage(image2)
	sim1.imshow()
	cv2.waitKey(0)


def test_exposure_correct():
	image1 = '../test_data/custom_db/0.jpg'
	image2 = '../test_data/custom_db/1.jpg'
	image3 = '../test_data/custom_db/2.jpg'

	# Read Images and detect keypoints
	print('Reading images and detecting keypoints')
	sim1 = SJPImage(image1)
	sim2 = SJPImage(image2)
	sim3 = SJPImage(image3)
	stitcher = Stitcher()
	sim1.setKeypoints(stitcher.getKeyPoints(sim1.getImage()))
	sim2.setKeypoints(stitcher.getKeyPoints(sim2.getImage()))
	sim3.setKeypoints(stitcher.getKeyPoints(sim3.getImage()))

	sim1.updateOverlappingRegions(None, sim2)
	sim2.updateOverlappingRegions(sim1, sim3)
	sim3.updateOverlappingRegions(sim2, None)

	# Prepare list of mean values for exposure correction
	mean_intensity_list = []
	mean_intensity_list.append(sim1.getImageOverlapMeans())
	mean_intensity_list.append(sim2.getImageOverlapMeans())
	mean_intensity_list.append(sim3.getImageOverlapMeans())
	print(mean_intensity_list)

	# Exposure correction
	ex_corrector = OptimizeExposure()
	ex_corrector.addImageIntensityPairs(mean_intensity_list)
	vals = ex_corrector.getMeanIntensities()
	gains = ex_corrector.calculateGains()
	print('Optimized gains: ', gains)
	sim2.setGain(gains[1])
	sim2.setNeighbourGains(gains[0], gains[2])
	# Exposure correct the middle image
	sim2.exposureCorrectJumpStyle()


def test_optical_flow():
	args = arg_setup()
	stitcher = Stitcher()
	image1 = args["first"]
	image2 = args["second"]

	# Read Images and detect keypoints
	print('Reading images and detecting keypoints')
	sim1 = SJPImage(image1)
	sim2 = SJPImage(image2)
	sim1.setKeypoints(stitcher.getKeyPoints(sim1.getImage()))
	sim2.setKeypoints(stitcher.getKeyPoints(sim2.getImage()))

	# Calculate Optical flow
	of_cal = OpticalFlowCalculator()
	flow = of_cal.calculateFlow(sim1.getImage(), sim2.getImage())


def main():
	# test_overlap()
	# test_SJPFramework()
	test_exposure_correct()
	# test_optical_flow()



if __name__ == '__main__':
	main()