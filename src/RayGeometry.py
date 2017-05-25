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

def normalizeX(x, width):
	return np.float32(x)/np.float32(width)

def normalizeY(y, height):
	return np.float32(y)/np.float32(height)

def normalizeXY(x, y, width, height):
	return normalizeX(x, width), normalizeY(y, height)

def unnormalizeX(xn, width):
	return xn*width

def unnormalizeY(yn, height):
	return yn*height

def unnormalizeXY(xn, yn, width, height):
	return unnormalizeX(xn, width), unnormalizeY(yn, height)

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

def degree2Radians(degree):
	return degree*np.pi/180


def getAngle(centre, cam_pos, ipd):
	hor = np.linalg.norm(centre-cam_pos)
	ver = ipd/2
	return np.arctan2(ver, hor)


def genPointOnViewingCircle(centre, cam_pos, ipd, eye=1):
	angle = getAngle(np.asarray(centre), np.asarray(cam_pos), ipd)
	switch_sign = False
	if cam_pos[1] > centre[1]:
		slope = (cam_pos[1]-centre[1])/(cam_pos[0]-centre[0])
	else:
		slope = (centre[1]-cam_pos[1])/(centre[0]-cam_pos[0])
		switch_sign = True
	perp_slope = -1/slope
	dir_vec = np.asarray([1, perp_slope], dtype='float32')
	dir_vec = unit_vector(dir_vec)
	if eye == 1:
		if switch_sign:
			res = centre - ipd*(dir_vec)
		else:
			res = centre + ipd*(dir_vec)
	else:
		if switch_sign:
			res = centre + ipd*(dir_vec)
		else:
			res = centre - ipd*(dir_vec)
	return res

def xzToTheta(xz, origin):
	vec = xz-origin
	return np.arctan2(vec[1], vec[0])

def mapCameraToSphere(camera_pos, origin, ipd, eye=1):
	point_on_circle = genPointOnViewingCircle(origin, camera_pos, ipd, eye=eye)
	# print(xzToTheta(point_on_circle, origin), radians2Degrees(xzToTheta(point_on_circle, origin)))
	# print(thetaToNormalizedX(xzToTheta(point_on_circle, origin)))
	return thetaToNormalizedX(xzToTheta(point_on_circle, origin))


def fitCircleTo3Points(point1, point2, point3):
	# Reference : http://paulbourke.net/geometry/circlesphere/
	m1 = (point2[1]-point1[1])/(point2[0]-point1[0])
	m2 = (point3[1]-point2[1])/(point3[0]-point2[0])

	centre_x = m1*m2*(point1[1]-point3[1]) + m2*(point1[0]+point2[0])
	centre_x = centre_x - m1*(point2[0]+point3[0])
	centre_x = centre_x/(2*(m2-m1))

	centre_y = (centre_x - ((point1[0]+point2[0])/2))*-1/m1
	centre_y = centre_y + ((point1[1]+point2[1])/2)

	return centre_x, centre_y


def frange(start, stop, step):
	i = start
	while i < stop:
		yield i
		i += step


def getCirclePoints(centre, radius, thresh=1e-4):
	xc = centre[0]
	yc = centre[1]
	circle_points = []
	step_size = np.float32(2*np.float32(radius)/100)
	for x in frange(xc-radius, xc+radius,step_size):
		for y in frange(yc-radius, yc+radius, step_size):
			tmp = (x-xc)**2 + (y-yc)**2
			tmp = np.abs(tmp - radius**2)
			if tmp <= thresh:
				circle_points.append([x, y])
	if (len(circle_points) <= 0):
		raise RuntimeError('Something bad with circle generation.')
	return np.asarray(circle_points, dtype='float32')




