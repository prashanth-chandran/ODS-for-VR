import numpy as np
import cv2
from SJPImage import *
cv2.ocl.setUseOpenCL(False)

def test_stitch():
	image1 = SJPImage(file_name= '../test_data/custom_db/ETH_HG/Polyterasse/0.jpg')
	image2 = SJPImage(file_name= '../test_data/custom_db/ETH_HG/Polyterasse/1.jpg')
	img_array = np.stack((image1.getImage(), image2.getImage()))
	print(img_array.shape)
	pano_creator = cv2.createStitcher(False)
	pano = pano_creator.stitch((image1.getImage(), image2.getImage()))
	cv2.imshow('res', pano[1])


def main():
	test_stitch()

if __name__ == '__main__':
	main()

