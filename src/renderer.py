import numpy as np 


def normalizedXYToThetaPhi(xn, yn):
	theta = (xn*2*np.pi)-np.pi
	phi = (np.pi/2) - (yn * np.pi)
	return theta, phi

def thetaPhiToNormalizedXY(theta, phi):
	xn = (theta + np.pi)/(2*np.pi)
	yn = ((np.pi/2) - phi)/np.pi
	return xn, yn


def normalizeXY(x, y, width, height):
	return np.float32(x)/np.float32(width), np.float32(y)/np.float32(height)

def unnormalizeXY(xn, yn, width, height):
	return xn*width, yn*height

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def getRayOrientation(ray):
	zaxis = np.asarray([0, 1], dtype='float32')
	yaxis = np.asarray([0, 1], dtype='float32')
	ray_xz = np.asarray([ray[0], ray[2]], dtype='float32')
	ray_yz = np.asarray([ray[1], ray[2]], dtype='float32')
	ray_xz = unit_vector(ray_xz)
	ray_yz = unit_vector(ray_yz)
	# theta = angle_between(ray_xz, zaxis)
	# phi = angle_between(ray_yz, yaxis)
	theta = np.arctan2(ray_xz[0], ray_xz[1])
	phi = np.arctan2(ray_yz[0], ray_yz[1])
	return theta, phi

def radians2Degrees(rad):
	return rad*180/np.pi

class Renderer():
	def __init__(self):
		pass

	def renderImage(self, image, camera, out_dim, nc=3):
		out_image = np.zeros((out_dim[0], out_dim[1], nc), dtype='uint8')
		image_dim = image.shape
		print(image_dim, out_dim)
		for x in range(image_dim[0]):
			for y in range(image_dim[1]):
				# xn, yn = normalizeXY(x, y, image_dim[0], image_dim[1])
				xn, yn = x, y
				ray = camera.getRayForPixel(xn, yn)
				ray = camera.transformRayToCameraRef(ray, camera.getExtrinsics())
				theta, phi = getRayOrientation(ray)
				# print(radians2Degrees(theta), radians2Degrees(phi))
				xn, yn = thetaPhiToNormalizedXY(theta, phi)
				xo, yo = unnormalizeXY(xn, yn, out_dim[0], out_dim[1])
				if xo < out_dim[0] and yo < out_dim[1]:
					out_image[np.int(xo), np.int(yo), :] = image[x, y, :]
					# print('Mapping ', x, ', ', y , ' to ', xo, ', ', yo)
				else:
					raise RuntimeError('Ray dimensions exceed output image dimension.')

		return out_image



