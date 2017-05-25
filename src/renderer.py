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


class RenderODS():
	def __init__(self):
		pass

		
	def linearInterpolation(theta_0, theta_1,theta_a, theta_b):
		diff_1b=theta_1-theta_b
		diff_0a=theta_0-theta_a
		diff_ba=theta_b-theta_a
		diff_01=theta_0-theta_1
		
		theta_p=((diff_1b*theta_0)+(diff_0a*theta_1))/(diff_ba+diff_01)

	def setCameraList(self, camera_collection):
		self.camera_list = camera_list





