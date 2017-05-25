import numpy as np

def normalizedXYToThetaPhi(xn, yn):
	theta = (xn*2*np.pi)-np.pi
	phi = (np.pi/2) - (yn * np.pi)
	return theta, phi

def thetaToNormalizedX(theta):
	return (theta + np.pi)/(2*np.pi)

def phiToNormalizedY(phi):
	return ((np.pi/2) - phi)/np.pi

def thetaPhiToNormalizedXY(theta, phi):
	xn = thetaToNormalizedX(theta)
	yn = phiToNormalizedY(phi)
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
	xaxis = np.asarray([1, 0, 0], dtype='float32')
	yaxis = np.asarray([0, 1, 0], dtype='float32')
	zaxis = np.asarray([0, 0, 1], dtype='float32')
	ray_xz = np.asarray([ray[0], ray[2]], dtype='float32')
	ray_yz = np.asarray([ray[1], ray[2]], dtype='float32')
	ray_xz = unit_vector(ray_xz)
	ray_yz = unit_vector(ray_yz)
	theta = np.arctan2(ray_xz[0], ray_xz[1])
	phi = np.arctan2(ray_yz[0], ray_yz[1])
	return theta, phi

def radians2Degrees(rad):
	return rad*180/np.pi


def getAngle(centre, cam_pos, ipd):
	hor = np.linalg.norm(centre-cam_pos)
	ver = ipd/2
	return np.arctan2(ver, hor)


def genPointOnCircle(centre, cam_pos,ipd):
	angle = getAngle(np.asarray(centre), np.asarray(cam_pos), ipd)
	slope = np.tan(angle)
	perp_slope = -1/slope
	dir_vec = np.asarray([1, perp_slope], dtype='float32')
	dir_vec = unit_vector(dir_vec)
	res = centre + ipd/2*(dir_vec)
	return res

def xzToTheta(xz, origin):
	xc = xz[0]
	zc = xz[1]
	return np.arctan2(zc, xc)

def getCameraOnSphere(camera_pos, origin, ipd):
	point_on_circle = genPointOnCircle(camera_pos, origin, ipd)
	return thetaToNormalizedX(xzToTheta(point_on_circle, origin))



