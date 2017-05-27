import numpy as np 
from cameras import *
from RayGeometry import *

class Renderer():
	def __init__(self):
		pass

	def renderImage(self, image, camera, out_dim, nc=3):
		out_image = np.zeros((out_dim[0], out_dim[1], nc), dtype='uint8')
		image_dim = image.shape
		print(image_dim, out_dim)
		row_list = [image_dim[0]/2]
		col_list = [image_dim[1]/2]
		for x in range(image_dim[0]):
			for y in range(image_dim[1]):
				# xn, yn = normalizeXY(x, y, image_dim[0], image_dim[1])
				xn, yn = x, y
				ray = camera.getRayForPixel(xn, yn)
				ray = camera.transformRayToCameraRef(ray, camera.getExtrinsics())
				theta, phi = getRayOrientation(ray)
				xn, yn = thetaPhiToNormalizedXY(theta, phi)
				xo, yo = unnormalizeXY(xn, yn, out_dim[0], out_dim[1])
				if xo < out_dim[0] and yo < out_dim[1]:
					out_image[np.int(xo), np.int(yo), :] = image[x, y, :]
				else:
					# raise RuntimeError('Ray dimensions exceed output image dimension.')
					# print('Mapping ', x, ', ', y , ' to ', xo, ', ', yo)
					pass

		return out_image


class RendererODS():
	def __init__(self):
		self.init_complete = False

		
	def linearInterpolation(theta_0, theta_1,theta_a, theta_b):
		diff_b1=theta_b-theta_1
		diff_0a=theta_0-theta_a
		diff_ba=theta_b-theta_a
		diff_01=theta_0-theta_1
		
		theta_p=((diff_b1*theta_0)+(diff_0a*theta_1))/(diff_ba+diff_01)

	def setCameraList(self, camera_collection):
		self.camera_list = camera_collection
		self.init_complete = True

	def setImageList(self, image_collection):
		self.image_list = image_collection

	def sanityCheck(self):
		if not self.init_complete:
			raise RuntimeError('Camera collection is not initialized')


	def renderCOPSOnly(self, ipd, output_image_dim, eye=-1, origin=[0, 0, 0]):
		self.sanityCheck()
		height = output_image_dim[0]
		width = output_image_dim[1]
		output_image = np.zeros((output_image_dim[0], output_image_dim[1], 3), dtype='uint8')

		camera_positions = self.camera_list.getCameraCentresXZ(origin)
		viewing_circle_centre = self.camera_list.getViewingCircleCentre()
		viewing_circle_radius = self.camera_list.getViewingCircleRadius()
		# IPD cannot be greater than the radius of the viewing circle
		if ipd > viewing_circle_radius:
			raise RuntimeError('IPD cannot be greater than the radius of the viewing circle')

		num_cameras = self.camera_list.getNumCameras()
		
		# Set COP for every camera and do other things before view interpolation.
		for i in range(num_cameras):
			theta = getAngle(viewing_circle_centre, camera_positions[i, :], ipd)
			self.camera_list[i].setCOPRelativeAngleLeft(theta)
			self.camera_list[i].setCOPRelativeAngleRight(theta)

			xn_left = mapCameraToSphere(camera_positions[i, :], viewing_circle_centre, ipd, -1)
			self.camera_list[i].setPositionInODSImageLeft(xn_left)
			xn_right = mapCameraToSphere(camera_positions[i, :], viewing_circle_centre, ipd, 1)
			self.camera_list[i].setPositionInODSImageRight(xn_right)

			if eye is 1:
				xn = xn_right
				col_img = self.camera_list[i].getCOPRight()
			else:
				xn = xn_left
				col_img = self.camera_list[i].getCOPLeft()

			col_index = int(unnormalizeX(xn, width))

			if self.image_list is None:
				output_image[:, col_index, :] = [127, 127, 127]
			else:
				# print(col_img)
				image = self.image_list[i]
				output_image[:, col_index, :] = image.getColumn(int(col_img))
				
		
		
		#view interpolation
		vertical_pixel=0
		for i in range(num_cameras):
			cam0=self.camera_list[i]
			cam1=self.camera_list[i+1]
			cam_position0=camera_positions[i, :]
			cam_position1=camera_positions[i+1, :]
			
			relative_theta_0=cam0.getCOPRelativeAngleLeft
			relative_theta_1=cam1.getCOPRelativeAngleLeft
			
#			x0=cam0.getCOPLeft(self)
#			tangent_pixel0=[x0, vertical_pixel]
#			theta_0=cam0.getGlobalAngleOfPixel(cam_position0, tangent_pixel0)
#			flow_0=computeOpticalFlow(tangent_pixel0)
#			tangent_pixel0_flow=[x0+flow_0[0], vertical_pixel]
#			theta_0_flow=cam0.getGlobalAngleOfPixel(cam_position0, tangent_pixel0_flow)
			
#			x1=cam1.getCOP(self)
#			tangent_pixel0=[x1, vertical_pixel]
#			theta_1=cam1.getGlobalAngleOfPixel(cam_position0, tangent_pixel1)
#			flow_1=computeOpticalFlow(tangent_pixel1)
#			tangent_pixel1_flow=[x1+flow_1[0], vertical_pixel]
#			theta_1_flow=cam1.getGlobalAngleOfPixel(cam_position1, tangent_pixel1_flow)
			
#			for j in range(tangent_pixel0:tangent_pixel1_flow):
#				p1=[j, vertical_pixel]
#				theta_a=cam0.getGlobalAngleOfPixel(cam_position0, p)
#				flow_i=computeOpticalFlow(p1)
#				p2=[j+flow_i[0], vertical_pixel]
#				theta_b=cam1.getGlobalAngleOfPixel(cam_position1, p)
#				
#				theta_p=linearInterpolation(theta_0, theta_1, theta_a, theta_b)
				


		return output_image



	def visualizeProjectionCentres(self, output_image_dim):
		self.sanityCheck()
		height = output_image_dim[0]
		width = output_image_dim[1]
		output_image = np.zeros((output_image_dim[0], output_image_dim[1]), dtype='uint8')
		
		camera_positions = self.camera_list.getCameraCentresXZ([0, 0, 0])
		viewing_circle_centre = self.camera_list.getViewingCircleCentre()
		viewing_circle_radius = self.camera_list.getViewingCircleRadius()
		# IPD for testing
		ipd = viewing_circle_radius/2
		num_cameras = self.camera_list.getNumCameras()
		skip_colors = int(255/num_cameras)
		
		for i in range(num_cameras):
			xn = mapCameraToSphere(camera_positions[i, :], viewing_circle_centre, ipd, eye=-1)
			col_index = int(unnormalizeX(xn, width))
			gray_scale = ((i+1)*skip_colors)
			output_image[:, col_index] = np.uint8(gray_scale)

		return output_image






