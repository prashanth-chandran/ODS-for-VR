import yaml
import numpy as np
import os


class Camera:
	def __init__(self):
		self.camera_name = None
		self.camera_overlaps = None
		self.intrinsics = np.zeros((3, 3), dtype='float32')
		self.extrinsics_relative = np.zeros((4,4), dtype='float32')
		# Current calibration file does not have absolute parameters
		# self.extrinsics_absolute = np.zeros((4,4), dtype='float32')
		self.distortion = np.zeros((4, 1), dtype='float32')
		self.resolution = np.zeros((2, 1), dtype='int32')
		self.fx = 0
		self.fy = 0
		self.favg = 0
		self.cam_collection = None
		self.init_complete = False

	def loadCameraFromYaml(self, yaml_calibration, cam_name):
		self.camera_name = cam_name
		self.resolution[0] = yaml_calibration[cam_name]['resolution'][0]
		self.resolution[1] = yaml_calibration[cam_name]['resolution'][1]
		
		intrinsics_vector = yaml_calibration[cam_name]['intrinsics']
		self.intrinsics[0][0] = intrinsics_vector[0]
		self.fx = intrinsics_vector[0]
		self.intrinsics[1][1] = intrinsics_vector[1]
		self.fy = intrinsics_vector[1]
		self.intrinsics[0][2] = intrinsics_vector[2]/self.resolution[0]
		self.intrinsics[1][2] = intrinsics_vector[3]/self.resolution[1]
		self.intrinsics[2][2] = 1
		self.favg = (self.fx + self.fy)/2
		self.intrinsics_inverse = np.linalg.inv(self.intrinsics)

		self.camera_overlaps = yaml_calibration[cam_name]['cam_overlaps']
		try:
			self.extrinsics_relative = yaml_calibration[cam_name]['T_cn_cnm1']
		except Exception as e:
			self.extrinsics_relative = np.identity(4, dtype='float32')
		self.distortion = yaml_calibration[cam_name]['distortion_coeffs']

		self.init_complete = True


	def getIntrinsics(self):
		self.cameraSanityCheck()
		return self.intrinsics

	def getExtrinsics(self):
		self.cameraSanityCheck()
		return self.extrinsics_relative

	def cameraSanityCheck(self):
		if self.init_complete is False:
			raise RuntimeError('Camera not initialized. Initialize camera before using it.')

	def getRayForPixel(self, x, y):
		self.cameraSanityCheck()
		pix_homo = np.asarray([x, y, 1], dtype='float32')
		ray = np.dot(self.intrinsics_inverse, pix_homo)
		return ray

	def transformRayToCameraRef(self, ray, camera_extrinsics):
		self.cameraSanityCheck()
		ray_homogeneous = np.append(ray, 1)
		return np.dot(camera_extrinsics, ray_homogeneous)[0:3]

# end class Camera 


def load_camera_calibration_data(file_name):
	with open(file_name, 'r') as stream:
		try:
			calib_data = yaml.load(stream)
		except Exception as e:
			raise RuntimeError('Something bad happened when loading the .yaml file')

		return calib_data

