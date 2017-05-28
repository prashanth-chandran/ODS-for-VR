from renderer import *
from cameras import *
from SJPImage import *
import argparse


def arg_setup():
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--first", required=True, help="path to image file")
	ap.add_argument("-s", "--second", required=True, help="path to calibration file")
	args = vars(ap.parse_args())
	return args

def test_renderer():
	args = arg_setup()
	img_name = args["first"]
	calib_file_name = args["second"]
	sjpimage = SJPImage(file_name=img_name)
	yaml_data = load_camera_calibration_data(calib_file_name)
	cam1 = Camera()
	cam1.loadCameraFromYaml(yaml_data, 'cam0')
	print(cam1.getIntrinsics(), cam1.getExtrinsics())
	r = Renderer()
	out_dim = [400, 800]
	out_image = r.renderImage(sjpimage.getImage(), cam1, out_dim)
	cv2.imshow('result', out_image)
	cv2.waitKey(0)


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

	# vis_image = rods.visualizeProjectionCentres([255, 800])
	# Rendering function from yesterday: 
	# vis_image = rods.renderCOPSOnly(0.045/2, [480, 960])

	# rods.renderGlobalVisTest(0.045/2, [480, 960])

	vis_image_right = rods.renderStuffBruh(0.045/2, [480, 960], eye=1)
	cv2.imshow('Projection centres right: ', vis_image_right)
	vis_image_left = rods.renderStuffBruh(0.045/2, [480, 960], eye=-1)
	cv2.imshow('Projection centres left: ', vis_image_left)
	cv2.waitKey(0)


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
	# test_renderer()
	test_ODS_renderer()
	# test_cameraRig_visualization()
	# test_data_loader()


if __name__ == '__main__':
	main()