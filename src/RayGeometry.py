import numpy as np

def normalizedXToTheta(X):
	return  (X*(2*np.pi))-np.pi 

def normalizedYToPhi(yn):
	phi = (np.pi/2) - (yn * np.pi)
	return phi

def normalizedXYToThetaPhi(xn, yn):
	"""
	convert normalized panaroma values to equivalent theta and phi values.
	"""
	theta = normalizedXToTheta(xn)
	phi = normalizedYToPhi(yn)
	return theta, phi

def thetaToNormalizedX(theta):
	return (theta + np.pi)/(2*np.pi)
	
def phiToNormalizedY(phi):
	return ((np.pi/2) + phi)/np.pi

def thetaPhiToNormalizedXY(theta, phi):
	"""
	Convert theta and Phi values to normalized x, y co-ordinates in the ODS stitch.
	"""
	xn = thetaToNormalizedX(theta)
	yn = phiToNormalizedY(phi)
	return xn, yn

def normalizeX(x, width):
	return np.float32(x)/np.float32(width)

def normalizeY(y, height):
	return np.float32(y)/np.float32(height)

def normalizeXY(x, y, width, height):
	"""
	Convert image co-ordinates to normalized image co-ordinates
	"""
	return normalizeX(x, width), normalizeY(y, height)

def unnormalizeX(xn, width):
	return xn*width

def unnormalizeY(yn, height):
	return yn*height

def unnormalizeXY(xn, yn, width, height):
	"""
	Convert normalized image co-ordinates to (floating point) pixel co-ordinates 
	"""
	return unnormalizeX(xn, width), unnormalizeY(yn, height)

def unit_vector(vector):
	"""
	Normalize the incoming vector and return a unit vector.
	"""
	return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
	"""
	Returns the angle between two vectors
	"""
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
	"""
	Convert an angle in radians to degrees (0 to +-180)
	"""
	return rad*180/np.pi

def degree2Radians(degree):
	"""
	Convert an angle in degrees to radians (-pi to +pi)
	"""
	return degree*np.pi/180

def radians2Degrees360(angle_rad):
	"""
	Convert an angle in radians to degrees ( 0 to 360)
	"""
	if angle_rad > 0:
		angle = angle_rad
	else:
		angle = 2*np.pi - np.abs(angle_rad)
	return radians2Degrees(angle)

def degrees3602Radians(degree):
	"""
	Converts degrees in 360 to radians
	"""
	rad = degree2Radians(degree)
	if rad >= np.pi:
		rad = -(2*np.pi - rad)
	return rad

def getAngle(centre, cam_pos, ipd):
	"""
	This function returns the angle between the centre of the viewing circle and the 
	vertical that drops from the camera onto the x-axis or the horizontal axis.
	"""
	hor = np.linalg.norm(centre-cam_pos)
	ver = ipd/2
	return np.arcsin(ver/hor)


def getPointOnVC(centre, point, ipd, eye=1):
	"""
	NOT USED:
	Given a 3D point, the centre of the viewing circle and the IPD, 
	this function returns a point that is tangential to the viewing circle and goes through the
	3D point.
	The function calculates the point on the viewing circle by rotating the vector that connects
	the centre and the point by an appropriate angle.
	"""
	dist1 = np.linalg.norm(np.asarray(point)-np.asarray(centre))
	theta = np.arcsin((ipd/2)/dist1)

	# Find the vector find the centre to the point
	vec = np.asarray(centre)-np.asarray(point)
	xv = vec[0]
	zv = vec[1]

	# Rotate this vector by theta for left eye and -theta for the right eye
	if eye==-1:
		x = xv*np.cos(theta) + zv*np.sin(theta)
		z = -xv*np.sin(theta) + zv*np.cos(theta)
	else:
		x = xv*np.cos(-theta) + zv*np.sin(-theta)
		z = -xv*np.sin(-theta) + zv*np.cos(-theta)

	vec[0] = x
	vec[1] = z
	# After rotation, normalize and scale the resulting vector.
	vec = unit_vector(vec)
	sidelen = np.sqrt(dist1**2 - (ipd/2)**2)
	vec = vec*sidelen
	# Displace 'point' by this vector to get the place where the tangent intersects the circle.
	# vec[0] and vec[1] contain the x and z co-ordinates of the point of intersection.
	vec = point + vec
	return vec

def getIntersectionOnVC(center, cam, point, radius, eye=1):
	"""
	NOT USED: Julia's complicated math stuff that I don't understand :D 
	"""
	m=(point[1]-cam[1])/(point[0]-cam[0])
	c=cam[1]-(m*cam[0])
	p=center[0]
	q=center[1]
	r=radius
	A=(m*m)+1
	B=2*((m*c)-(m*q)-p)
	C=(q*q)-(r*r)+(p*p)-(2*c*q)+(c*c)
	x1=(-B+np.sqrt((B*B)-(4*A*C)))/(2*A)
	x2=(-B-np.sqrt((B*B)-(4*A*C)))/(2*A)
	y1=m*x1+c
	y2=m*x2+c
	p1=x1, y1
	p2=x2, y2
	d1=np.linalg.norm(cam-p1)
	d2=np.linalg.norm(cam-p2)
	if d1<d2:
		return np.asarray(p1)
	else:
		return np.asarray(p2)
	

def xzToTheta(xz, origin):
	"""
	Find the angle in radians for some point in space.
	"""
	vec = xz-origin
	return np.arctan2(vec[1], vec[0])


def mapPointToODSAngle(point, origin, ipd, eye=1):
	dist = np.linalg.norm(np.asarray(point)-np.asarray(origin))
	r = ipd/2
	theta = np.arccos(r/dist)
	angle_to_x = xzToTheta(np.asarray(point), np.asarray(origin))
	# print('interior angle: ', radians2Degrees360(theta))
	# print('additive angle: ', radians2Degrees360(angle_to_x))
	theta_final = 0
	if eye==1:
		theta_final = np.mod(angle_to_x + theta - np.pi, 2*np.pi)
	else:
		theta_final = angle_to_x + theta

	return theta_final

def mapPointToODSColumn(point, origin, ipd, eye=1):
	global_theta = mapPointToODSAngle(point, origin, ipd, eye)
	# print(radians2Degrees360(global_theta))
	xn = thetaToNormalizedX(global_theta)
	# needs handling of discontinuity across 0 and 360 degrees.
	if xn > 1:
		xn = xn-1
	return xn


def get2DPointOnODSVC(point, origin, ipd, eye=1):
	t_final = mapPointToODSAngle(point, origin, ipd, eye)
	# print(radians2Degrees(t_final))
	x = (ipd/2)*np.cos(t_final)
	y = (ipd/2)*np.sin(t_final)
	x = x + origin[0]
	y = y + origin[1]
	point = np.asarray((x, y), dtype='float32')
	return point


def fitCircleTo3Points(point1, point2, point3):
	"""
	Given three points, this function estimates the centre and radius of a circle 
	that will pass through the points.
	"""
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
	"""
	Python does not have a default range() function for floating point numbers. This is a
	naive implementation.
	"""
	i = start
	while i < stop:
		yield i
		i += step


def getCirclePoints(centre, radius, thresh=1e-5):
	"""
	Given a centre and radius, this function returns points that lie on the circle.
	Adjust the value of thresh for a finer or a coarser circle.
	"""
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




