from renderer import *
from cameras import *
from SJPImage import *
import argparse


def arg_setup():
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--first", required=True, help="path to image")
	ap.add_argument("-s", "--second", required=True, help="path to camera calibration file")
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


def test_camera_visualization():
	args = arg_setup()
	cam_file = args["second"]
	yaml_data = load_camera_calibration_data(cam_file)
	cc = CameraCollection()
	cc.readAllCameras(cam_file)
	cc.visualizeCameras([0, 0, 0])


def main():
	# test_renderer()
	test_camera_visualization()


if __name__ == '__main__':
	main()