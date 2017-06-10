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
	image1 = '../test_data/custom_db/ETH_HG/Polyterasse/0.jpg'
	image2 = '../test_data/custom_db/ETH_HG/Polyterasse/1.jpg'
	image3 = '../test_data/custom_db/ETH_HG/Polyterasse/2.jpg'
	image4 = '../test_data/custom_db/ETH_HG/Polyterasse/3.jpg'
	image5 = '../test_data/custom_db/ETH_HG/Polyterasse/4.jpg'
	image6 = '../test_data/custom_db/ETH_HG/Polyterasse/5.jpg'
	image7 = '../test_data/custom_db/ETH_HG/Polyterasse/6.jpg'
	image8 = '../test_data/custom_db/ETH_HG/Polyterasse/7.jpg'
	sim1 = SJPImage(file_name=image1)
	sim2 = SJPImage(file_name=image2)
	sim3 = SJPImage(file_name=image3)
	sim4 = SJPImage(file_name=image4)
	sim5 = SJPImage(file_name=image5)
	sim6 = SJPImage(file_name=image6)
	sim7 = SJPImage(file_name=image7)
	sim8 = SJPImage(file_name=image8)

	sim1.initializeImage(sim8, sim2)
	sim2.initializeImage(sim1, sim3)
	sim3.initializeImage(sim2, sim4)
	sim4.initializeImage(sim3, sim5)
	sim5.initializeImage(sim4, sim6)
	sim6.initializeImage(sim5, sim7)
	sim7.initializeImage(sim6, sim8)
	sim8.initializeImage(sim7, sim1)

	pan_size = (500, 225)
	H = sim2.getHomographyLeft()
	HI = np.identity(3)
	temp0 = cv2.warpPerspective(sim1.getImage(), HI, pan_size)
	cv2.waitKey(0)

def test_SJP_Sequential():
	image1 = '../test_data/custom_db/ETH_HG/Polyterasse/0.jpg'
	image2 = '../test_data/custom_db/ETH_HG/Polyterasse/1.jpg'
	image3 = '../test_data/custom_db/ETH_HG/Polyterasse/2.jpg'
	image4 = '../test_data/custom_db/ETH_HG/Polyterasse/3.jpg'
	image5 = '../test_data/custom_db/ETH_HG/Polyterasse/4.jpg'
	image6 = '../test_data/custom_db/ETH_HG/Polyterasse/5.jpg'
	image7 = '../test_data/custom_db/ETH_HG/Polyterasse/6.jpg'
	image8 = '../test_data/custom_db/ETH_HG/Polyterasse/7.jpg'
	sim1 = SJPImage(file_name=image1)
	sim2 = SJPImage(file_name=image2)
	sim3 = SJPImage(file_name=image3)
	sim4 = SJPImage(file_name=image4)
	sim5 = SJPImage(file_name=image5)
	sim6 = SJPImage(file_name=image6)
	sim7 = SJPImage(file_name=image7)
	sim8 = SJPImage(file_name=image8)

	# define the size of the panaroma
	pan_size = (1200, 225)
	HI = np.identity(3)
	res1 = cv2.warpPerspective(sim1.getImage(), HI, pan_size)

	sres1 = SJPImage(image=res1.copy(), resize=False)
	sres1.initializeImage(None, sim2)
	sim2.initializeImage(sres1, None)
	H = sim2.getHomographyLeft()
	# H[2, 0 ]= 0
	# H[2, 1] = 0
	res2 = cv2.warpPerspective(sim2.getImage(), H, pan_size)
	
	sres2 = SJPImage(res2.copy(), resize=False)
	sres2.initializeImage(None, sim3)
	sim3.initializeImage(sres2, None)
	H = sim3.getHomographyLeft()
	# H[2, 0 ]=0
	# H[2, 1] = 0
	res3 = cv2.warpPerspective(sim3.getImage(), H, pan_size)

	sres3 = SJPImage(image=res3.copy(), resize=False)
	sres3.initializeImage(None, sim4)
	sim4.initializeImage(sres3, None)
	H = sim4.getHomographyLeft()
	# H[2, 0 ]=0
	# H[2, 1] = 0
	res4 = cv2.warpPerspective(sim4.getImage(), H, pan_size)

	sres4 = SJPImage(image=res4.copy(), resize=False)
	sres4.initializeImage(None, sim5)
	sim5.initializeImage(sres4, None)
	H = sim5.getHomographyLeft()
	# H[2, 0 ] = 0
	# H[2, 1] = 0
	res5 = cv2.warpPerspective(sim5.getImage(), H, pan_size)

	sres5 = SJPImage(image=res5.copy(), resize=False)
	sres5.initializeImage(None, sim6)
	sim6.initializeImage(sres5, None)
	H = sim6.getHomographyLeft()
	# H[2, 0 ]=0
	# H[2, 1] = 0
	res6 = cv2.warpPerspective(sim6.getImage(), H, pan_size)	


	pan_stack = np.stack((res1, res2, res3, res4, res5, res6), axis=-1)
	print(pan_stack.shape)
	res_avg = np.mean(pan_stack, axis=3)
	res_avg = np.uint8(res_avg)
	sres_avg = SJPImage(res_avg, resize=False)
	
	sres1.imshow('1')
	# sres2.imshow('2')
	# sres3.imshow('3')
	sres_avg.imshow('composite')
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
	test_SJP_Sequential()
	# test_exposure_correct()
	# test_optical_flow()



if __name__ == '__main__':
	main()