from Stitcher import *
import imutils
import cv2
import argparse


def testImageStitcher():
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--first", required=True, help="path to the first image")
	ap.add_argument("-s", "--second", required=True, help="path to the second image")
	ap.add_argument("-r", "--result", required=False, help="path to save the panaroma")
	args = vars(ap.parse_args())

	# Image IO
	image1 = cv2.imread(args["first"])
	print(args["first"], args["second"])
	image2 = cv2.imread(args["second"])
	# image1 = imutils.resize(image1, width=400)
	# image2 = imutils.resize(image2, width=400)
	print(image1.shape, image2.shape)

	# Call our stitch function
	stitcher = Stitcher()
	(pan, vis) = stitcher.stitch(image2, image1, showMatches=True)
	# Display routines
	cv2.imshow("Image 1", image1)
	cv2.imshow("Image 2", image2)
	cv2.imshow("Key point matches", vis)
	cv2.imshow("Stitched result", pan)
	cv2.waitKey(0)
	cv2.imwrite(args["result"], pan)


def buildPanaroma():
	im1 = cv2.imread('test_data/custom_db/0.jpg')
	im2 = cv2.imread('test_data/custom_db/1.jpg')
	im3 = cv2.imread('test_data/custom_db/2.jpg')
	im4 = cv2.imread('test_data/custom_db/3.jpg')

	im1 = imutils.resize(im1, width=400)
	im2 = imutils.resize(im2, width=400)
	im3 = imutils.resize(im3, width=400)
	im4 = imutils.resize(im4, width=400)

	st = Stitcher()
	(t1,v1) = st.stitch(im1, im2,  showMatches=True)
	(t2, v2) = st.stitch(im2, im3,  showMatches=True)
	(t3, v3) = st.stitch(im3, im4,  showMatches=True)
	cv2.imshow("Result 1", t1)
	cv2.imshow("Feat vis 1", v1)
	cv2.imshow("Result 2", t2)
	cv2.imshow("Feat vis 2", v2)
	cv2.imshow("Result 3", t3)
	cv2.imshow("Feat vis 3", v3)
	cv2.waitKey(0)


def main():
	testImageStitcher()
	# buildPanaroma()


if __name__ == '__main__':
	main()