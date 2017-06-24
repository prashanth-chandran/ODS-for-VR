from renderer import *
from cameras import *
from SJPImage import *
import matplotlib.pyplot as pyplt
import argparse
import scipy.misc


def arg_setup():
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--first", required=True, help="path to image file")
	ap.add_argument("-s", "--second", required=True, help="path to calibration file")
	args = vars(ap.parse_args())
	return args

def test_cameraRig_visualization():
	args = arg_setup()
	cam_file = args["second"]
	yaml_data = load_camera_calibration_data(cam_file)
	cc = CameraCollection()
	cc.readAllCameras(cam_file)
	print(cc[0].getIncidentColumn(degree2Radians(0)))
	for i in range(cc.getNumCameras()):
		print(cc[i].getFieldOfViewInDegrees())
	cc.visualizeCameras([0, 0, 0])


def test_ODS_renderer():
	args = arg_setup()
	cam_file = args["second"]
	image_file = args["first"]

	cc = CameraCollection()
	cc.readAllCameras(cam_file)
	ic0 = SJPImageCollection()
	ic0.loadImagesFromYAML(image_file, 'frame0')

	rods = RendererODS()
	rods.setImageList(ic0)
	rods.setCameraList(cc)

	pan_for_eye = rods.render360WithViewInterpolation(0.062, [480, 2000], eye=-1)
	pyplt.subplot(211), pyplt.imshow(pan_for_eye), pyplt.axis('off'), pyplt.title('360 - Left eye')
	pan_for_eye = rods.render360WithViewInterpolation(0.062, [480, 2000], eye=1)
	pyplt.subplot(212), pyplt.imshow(pan_for_eye), pyplt.axis('off'), pyplt.title('360 - Right eye')

	# This function goes over every camera in the list and plots image planes in the global frame. 
	# These points are also mapped onto the viewing circle.
	# rods.rigVisTest(0.062, [480, 960], eye=1)

	# vis_image_right = rods.render360NoInterpolation(0.062, [480, 2000], eye=1)
	# pyplt.subplot(211), pyplt.imshow(vis_image_right), pyplt.axis('off'), pyplt.title('360 - No interpolation - Left eye')
	# vis_image_left = rods.render360NoInterpolation(0.062, [480, 2000], eye=-1)
	# pyplt.subplot(212), pyplt.imshow(vis_image_left), pyplt.axis('off'), pyplt.title('360 - No interpolation - Right eye eye')
	# scipy.misc.imsave('fifth_stitch.jpg', vis_image_left)
	pyplt.show()


def test_data_loader():
	args = arg_setup()
	calib_file = args["second"]
	image_file = args["first"]
	cc = CameraCollection()
	cc.readAllCameras(calib_file)

	ic0 = SJPImageCollection()
	ic0.loadImagesFromYAML(image_file, 'frame0')
	ic0.getNumberOfImages()




def main():
	test_ODS_renderer()
	# test_cameraRig_visualization()
	# test_data_loader()


if __name__ == '__main__':
	main()