import yaml
import numpy as np
import os
import matplotlib.pyplot as pyplt


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
		ray=np.dot(camera_extrinsics, ray_homogeneous)
		#normalize ray again
		return ray[0:3]/ray[3]

# end class Camera 


class CameraCollection():

	def __init__(self):
		self.camera_collection = []
		self.num_cameras = 0
		self.init_complete = False

	def sanityCheck(self):
		if not self.init_complete:
			raise RuntimeError('Camera collection is not initialized')

	def addCamera(self, camera):
		self.camera_collection.append(camera)
		self.num_cameras = self.num_cameras + 1


	def readAllCameras(self, yaml):
		calib_data = load_camera_calibration_data(yaml)
		num_cameras = len(calib_data)

		for i in range(num_cameras):
			cam_name = 'cam' + str(i)
			c = Camera()
			c.loadCameraFromYaml(calib_data, cam_name)
			self.addCamera(c)


	def __getitem__(self, key):
		return self.camera_collection[key]


	def __len__(self):
		return self.num_cameras


	def visualizeCameras(self, origin):
		
		if len(origin) != 3:
			raise RuntimeError('Only 3D co-ordinates supported. Origin must be a 3D vector.')

		camera_xz_locs = np.zeros((self.num_cameras, 2), dtype='float32')
		# First camera is considered to be at the origin by default
		camera_xz_locs[0, :] = [origin[0], origin[2]]
		ordering=[0,1,2,3,8,9,7,6,4,5]
		for i in range(1, self.num_cameras):
			#ref = np.asarray([0, 0], dtype='float32')
			cam_i=ordering[i]
			cam_i_old=ordering[i-1]
			ref_cam = self.camera_collection[i-1].getExtrinsics()
			#ref_cam = self.camera_collection[cam_i_old].getExtrinsics()
			#ref = [ref_cam[0][3], ref_cam[2][3]]
			R=np.zeros((3, 3), dtype='float32')
			t=np.zeros((3, 1), dtype='float32')
			#print(ref_cam[0][0])
			
			#VERY ugly matrix assignment, I am sorry, I didn't find a nicer way :(
			R[0][0]=ref_cam[0][0]
			R[1][0]=ref_cam[1][0]
			R[2][0]=ref_cam[2][0]
			R[0][1]=ref_cam[0][1]
			R[1][1]=ref_cam[1][1]
			R[2][1]=ref_cam[2][1]
			R[0][2]=ref_cam[0][2]
			R[1][2]=ref_cam[1][2]
			R[2][2]=ref_cam[2][2]
			R=np.transpose(R)
			#print(R)
			t[0]=ref_cam[0][3]
			t[1]=ref_cam[1][3]
			t[2]=ref_cam[2][3]
			#print(t)
			ref=np.dot(-R,t)
			#print(ref)
			#current_camera_relative_loc = self.camera_collection[i].getExtrinsics()
			cam = self.camera_collection[i].getExtrinsics()
			#ref = [ref_cam[0][3], ref_cam[2][3]]
			R2=np.zeros((3, 3), dtype='float32')
			t2=np.zeros((3, 1), dtype='float32')
			#print(ref_cam[0][0])
			
			#VERY ugly matrix assignment, I am sorry, I didn't find a nicer way :(
			R2[0][0]=cam[0][0]
			R2[1][0]=cam[1][0]
			R2[2][0]=cam[2][0]
			R2[0][1]=cam[0][1]
			R2[1][1]=cam[1][1]
			R2[2][1]=cam[2][1]
			R2[0][2]=cam[0][2]
			R2[1][2]=cam[1][2]
			R2[2][2]=cam[2][2]
			R2=np.transpose(R2)
			#print(R)
			t2[0]=cam[0][3]
			t2[1]=cam[1][3]
			t2[2]=cam[2][3]
			#print(t)
			pos=np.dot(-R2,t2)
			#camera_xz_locs[i, :] = ref[0]+current_camera_relative_loc[0][3], ref[2]+current_camera_relative_loc[2][3]
			camera_xz_locs[cam_i, :] =ref[0]+pos[0], ref[2]+pos[2]
			
		fig, ax = pyplt.subplots()
		ax.scatter(camera_xz_locs[:, 0], camera_xz_locs[:, 1])
		for i in range(self.num_cameras):
			ax.annotate(self.camera_collection[i].camera_name, (camera_xz_locs[i, 0], camera_xz_locs[i, 1]))
		pyplt.show()



def load_camera_calibration_data(file_name):
	with open(file_name, 'r') as stream:
		try:
			calib_data = yaml.load(stream)
		except Exception as e:
			raise RuntimeError('Something bad happened when loading the .yaml file')

		return calib_data

