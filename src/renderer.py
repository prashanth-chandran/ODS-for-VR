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


	def setCameraList(self, camera_collection):
		self.camera_list = camera_collection
		self.init_complete = True

	def sanityCheck(self):
		if not self.init_complete:
			raise RuntimeError('Camera collection is not initialized')


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
			xn = mapCameraToSphere(camera_positions[i, :], viewing_circle_centre, ipd, eye=1)
			col_index = int(unnormalizeX(xn, width))
			gray_scale = ((i+1)*skip_colors)
			output_image[:, col_index] = np.uint8(gray_scale)

		return output_image






