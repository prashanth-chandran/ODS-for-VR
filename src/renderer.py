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


	def setCameraList(self, camera_collection):
		self.camera_list = camera_list





