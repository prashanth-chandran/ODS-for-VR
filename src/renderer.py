import numpy as np 

from cameras import *
from RayGeometry import *
from viewSynth import *

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

		
	def linearInterpolation(self, theta_0, theta_1,theta_a, theta_b):
		#print("theta_0")
		#print(theta_0)
		#print("theta_1")
		#print(theta_1)
		
		#print("theta_a")
		#print(theta_a)
		#print("theta_b")
		#print(theta_b)

		diff_b1=theta_1-theta_b
		diff_0a=theta_0-theta_a
		diff_ba=theta_b-theta_a
		diff_10=theta_0-theta_1
		
		theta_p=((diff_b1*theta_0)+(diff_0a*theta_1))/(diff_ba+diff_10)
		
		return theta_p
	
	def ourLinearInterpolation(self, rtheta_0, rtheta_1, rtheta_a, rtheta_b, theta_0, theta_1):
		diff_b1=rtheta_b-rtheta_1
		diff_0a=rtheta_0-rtheta_a
		diff_ba=rtheta_b-rtheta_a
		diff_01=rtheta_0-rtheta_1
		
		diff=np.abs(theta_0-theta_1)
		
		theta_p=((diff_b1*theta_0)+(diff_0a*theta_1))/(diff_ba+diff_01)
		#theta_p=((diff_b1*diff)+(diff_0a*diff))/(diff_ba+diff_01)
		while theta_p<-np.pi:
			theta_p=theta_p+np.pi
			
		theta_p=np.pi+theta_p
			
		return theta_p
	
	def ourInterpolation(self, rtheta_0, rtheta_1, rtheta_a, rtheta_b, theta_0, theta_1):
		factor1=(rtheta_a-rtheta_0)/rtheta_0
		factor2=(rtheta_b-rtheta_1)/rtheta_1
		theta_p =(((1+factor1)*theta_0)+((1-factor2)*theta_1))/((factor1+factor2))
		return theta_p
		
	def myInterpolation(self, rtheta_0, rtheta_1, rtheta_a, rtheta_b, theta_0, theta_1, rtheta_0_cam1, rtheta_1_cam0):
		factor1=np.abs((rtheta_a-rtheta_0)/(rtheta_1_cam0-rtheta_0))
		factor2=np.abs((rtheta_b-rtheta_1)/(rtheta_0_cam1-rtheta_1))
		#factor1=np.abs((rtheta_a-rtheta_0)/(rtheta_0-rtheta_1_cam0))
		#factor2=np.abs((rtheta_b-rtheta_1)/(rtheta_0_cam1-rtheta_1))
		
		#theta_p =(((1+factor1)*(theta_1-theta_0))+((1-factor2)*(theta_1-theta_0)))/((factor1+factor2))
		#theta_p=(((factor2*(theta_0-theta_1))+theta_1)+(theta_0-(factor1*(theta_0-theta_1))))/(theta_0-theta_1)
		
		factor3=factor1*(theta_0-theta_1)
		factor4=factor2*(theta_0-theta_1)
		
		diff=theta_0+theta_1
		abs=np.abs(diff)
		#theta_p=((theta_1+factor3)+(theta_1+factor4))/(factor2+factor1)	#theta_p=((theta_0-(factor1*diff))+((factor2*diff)+theta_1))/(((factor1*diff)+factor2*diff))
		#theta_p=(((1-factor3)*theta_0)+((factor4*theta_1)))/(factor3+factor4)
		#theta_p=theta_1+(factor2*diff)
		theta_p=theta_0-(factor1*abs)
		
		while theta_p<-np.pi:
			theta_p=theta_p+np.pi
			
		theta_p=np.pi+theta_p
			
		
		return theta_p


	def interpolateLocal(self, cop_column0, cop_angle, pix_column0, width, fov, t0, pan_width):
		add = True
		if pix_column0 > width/2:
			pix_c = pix_column0 - width/2
		else:
			add = False
			pix_c = width/2 - pix_column0

		delta = np.arctan(2*np.float32(pix_c)/np.float32(width) * np.tan(fov/2))

		if not add:
			D = fov/2 + delta
			T = fov/2 - cop_angle
			contrib1 = - (D-T)
		else:
			contrib1 = - (cop_angle - delta)

		ncontrib1 = contrib1/fov
		
		g_skip = degree2Radians(pan_width/360.0)
		contrib1 = ncontrib1*g_skip
		avg = (contrib1+t0)
		return avg

	def normalizeThenInterpolate(self, t0, t1, ta, tb):
		# handle special case of the discontinuity between 0 to 360 degrees
		if abs(t0-t1) <= 270.0:
			drange = t1-t0
			t0s = (t0-t0)/drange
			t1s = (t1-t0)/drange
			tas = (ta-t0)/drange
			tbs = (tb-t0)/drange
			tps = self.linearInterpolation(t0s, t1s, tas, tbs)
			tps = tps*drange + t0
		else:
			t1_new = abs(t1-360)+t0
			tb_new = abs(tb-360)+t0
			drange = abs(t1_new)
			t0s = (t0-t0)/drange
			t1s = (abs(t1_new))/drange
			tas = (t0-ta)/drange
			tbs = (abs(tb_new))/drange
			tps = self.linearInterpolation(t0s, t1s, tas, tbs)
			tps = t0-(tps*drange)+360

		np.clip(tps, 0.0, 1.0)
		return tps

		
	def weightedAverage(self, rtheta_0, rtheta_1, rtheta_a, rtheta_b, theta_0, theta_1):
		f = (rtheta_a/rtheta_0)*theta_0
		s = (1-(rtheta_b/rtheta_1))*theta_1
		theta_p=(f+s)/2
		
		while theta_p<-np.pi:
			theta_p=theta_p+np.pi
			
		theta_p=np.pi+theta_p
		
		return theta_p

	def setCameraList(self, camera_collection):
		self.camera_list = camera_collection
		self.init_complete = True

	def setImageList(self, image_collection):
		self.image_list = image_collection

	def sanityCheck(self):
		if not self.init_complete:
			raise RuntimeError('Camera collection is not initialized')


	def renderGlobalVisTest(self, ipd, out_image_dim, eye=1, origin=[0, 0, 0]):
		self.sanityCheck()
		# NOTE: ALL RENDERING IN THIS TEST FUNCTION IS DONE FOR THE RIGHT EYE!
		height = out_image_dim[0]
		width = out_image_dim[1]
		output_image = np.zeros((out_image_dim[0], out_image_dim[1], 3), dtype='uint8')

		camera_positions = self.camera_list.getCameraCentresXZ(origin)
		viewing_circle_centre = self.camera_list.getViewingCircleCentre()
		viewing_circle_radius = self.camera_list.getViewingCircleRadius()

		if ipd > viewing_circle_radius:
			raise RuntimeError('IPD cannot be greater than the radius of the viewing circle')

		nc = self.camera_list.getNumCameras()
		for i in range(0, nc):
			theta = getAngle(viewing_circle_centre, camera_positions[i, :], ipd)
			self.camera_list[i].setCOPRelativeAngleRight(theta)
			xn_right = mapCameraToSphere(camera_positions[i, :], viewing_circle_centre, ipd, 1)
			self.camera_list[i].setPositionInODSImageRight(xn_right)

			if eye is 1:
				xn = xn_right
				col_img = self.camera_list[i].getCOPRight()
			else:
				xn = xn_left
				col_img = self.camera_list[i].getCOPLeft()

			col_index = int(unnormalizeX(xn, width))
			# print(col_index)

			# Rays for the start, middle and end of an image
			ray = self.camera_list[i].getRayForPixel(self.camera_list[i].resolution[0]/2, 0)
			ray_start = self.camera_list[i].getRayForPixel(0, 0)
			ray_end = self.camera_list[i].getRayForPixel(self.camera_list[i].resolution[0], 0)
			# Normalize rays and convert them into 4D homogenous co-ordinates
			ray = unit_vector(ray)
			ray_start = unit_vector(ray_start)
			ray_end = unit_vector(ray_end)
			ray = np.append(ray, 1)
			ray_start = np.append(ray_start, 1)
			ray_end = np.append(ray_end, 1)
			# Transfer rays onto the global co-ordinate frame
			point_in_3d = np.dot(self.camera_list[i].extrinsics_absolute, ray)
			point_in_3d_start = np.dot(self.camera_list[i].extrinsics_absolute, ray_start)
			point_in_3d_end = np.dot(self.camera_list[i].extrinsics_absolute, ray_end)
			# Find where these rays map onto the ODS panaroma
			xn_new = mapCameraToSphere((point_in_3d[0], point_in_3d[2]), viewing_circle_centre, ipd, 1)
			xn_new_start = mapCameraToSphere((point_in_3d_start[0], point_in_3d_start[2]), viewing_circle_centre, ipd, 1)
			xn_new_end = mapCameraToSphere((point_in_3d_end[0], point_in_3d_end[2]), viewing_circle_centre, ipd, 1)

			# point = genPointOnViewingCircle(viewing_circle_centre, camera_positions[i, :], ipd, eye=1)
			point = getPointOnVC(viewing_circle_centre, np.asarray([point_in_3d[0], point_in_3d[2]]), ipd)
			point2 = getPointOnVC(viewing_circle_centre, np.asarray([point_in_3d_start[0], point_in_3d_start[2]]), ipd)
			point3 = getPointOnVC(viewing_circle_centre, np.asarray([point_in_3d_end[0], point_in_3d_end[2]]), ipd)
			# print(xn_right, xn_new)
			# print(xn_new_start, xn_new, xn_new_end)

			# Plots, plots and more plots
			fig, ax = pyplt.subplots()
			# Plot ray in camera's local frame of reference
			ax.scatter(ray[0], ray[2])
			ax.annotate('R', (ray[0], ray[2]))

			# Plot cameras in XZ. Current camera is highlighted in green. All other cameras are red.
			ax.annotate('C', (camera_positions[i, 0], camera_positions[i, 1]), color='green')
			for k in range(0, nc):
				ax.scatter(camera_positions[k, 0], camera_positions[k, 1], color='red')
			ax.scatter(camera_positions[i, 0], camera_positions[i, 1], color='green')
			ax.hold('on')

			# Plot where the start, middle and end of an image are mapped in the global frame of reference
			ax.scatter(point_in_3d[0], point_in_3d[2], color='green')
			ax.annotate('M', (point_in_3d[0], point_in_3d[2]))
			ax.scatter(point_in_3d_start[0], point_in_3d_start[2], color='green')
			ax.annotate('S', (point_in_3d_start[0], point_in_3d_start[2]))
			ax.scatter(point_in_3d_end[0], point_in_3d_end[2], color='green')
			ax.annotate('E', (point_in_3d_end[0], point_in_3d_end[2]))

			# Plot where the start, middle and end of an image are mapped in the viewing circle.
			ax.scatter(point[0], point[1], color='green')
			ax.annotate('M', (point[0], point[1]))
			ax.scatter(point2[0], point2[1], color='yellow')
			ax.annotate('S', (point2[0], point2[1]))
			ax.scatter(point3[0], point3[1], color='yellow')
			ax.annotate('E', (point3[0], point3[1]))
			
			# Plot origin and the viewing circle centre.
			ax.scatter(0, 0)
			ax.annotate('O', (0, 0))
			ax.annotate('VC', (viewing_circle_centre[0], viewing_circle_centre[1]))
			ax.scatter(viewing_circle_centre[0], viewing_circle_centre[1], color='blue')

			# Plot viewing circle
			viewing_circle = getCirclePoints(viewing_circle_centre, ipd/2)
			ax.scatter(viewing_circle[:, 0],viewing_circle[:, 1], color='red')
			pyplt.show()
			cn = int(unnormalizeX(xn_new, width))
			print('camera ', i, ': ', cn)


	def renderStuffBruh(self, ipd, out_image_dim, eye=1, origin=[0, 0, 0]):
		self.sanityCheck()
		height = out_image_dim[0]
		pan_width = out_image_dim[1]
		output_image = np.zeros((out_image_dim[0], out_image_dim[1], 3), dtype='uint8')
		nc = self.camera_list.getNumCameras()
		image_stacker = np.zeros((out_image_dim[0], out_image_dim[1], 3, nc), dtype='uint8')

		camera_positions = self.camera_list.getCameraCentresXZ(origin)
		viewing_circle_centre = self.camera_list.getViewingCircleCentre()
		viewing_circle_radius = self.camera_list.getViewingCircleRadius()

		if ipd > viewing_circle_radius:
			raise RuntimeError('IPD cannot be greater than the radius of the viewing circle')

		for i in range(0, nc):
			theta = getAngle(viewing_circle_centre, camera_positions[i, :], ipd)
			self.camera_list[i].setCOPRelativeAngleRight(theta)
			self.camera_list[i].setCOPRelativeAngleLeft(theta)
			
			xn_right = mapCameraToSphere(camera_positions[i, :], viewing_circle_centre, ipd, eye=1)
			xn_left = mapCameraToSphere(camera_positions[i, :], viewing_circle_centre, ipd, eye=-1)
			self.camera_list[i].setPositionInODSImageRight(xn_right)
			self.camera_list[i].setPositionInODSImageLeft(xn_left)

			if eye is 1:
				xn = xn_right
			else:
				xn = xn_left

			col_index = int(unnormalizeX(xn, pan_width))
			# Map this camera to an angle on the viewing circle.

			cam_theta = mapPointToPanaromaAngle(camera_positions[i, :], viewing_circle_centre, ipd, eye)

			print('camera ', i, 'maps to: ', 'angle: ', radians2Degrees(cam_theta), 'column: ',  col_index)
			
			print('Rendering camera ', i)
			# Temporary image that holds the render result just for this camera
			temp_image = np.zeros((out_image_dim[0], out_image_dim[1], 3), dtype='uint8')
			# Go over all columns in the image and get a ray for each column.
			print("resolution")
			print(int(self.camera_list[i].resolution[0]))
			for col in range(0, int(self.camera_list[i].resolution[0])):
				# Setting row index to zero, because it doesn't really matter
				ray = self.camera_list[i].getRayForPixel(col, 0)
				# Normalize and convert ray to 4D homogenous co-ordinates
				ray = unit_vector(ray)
				ray = np.append(ray, 1)
				# Transfer this ray into global co-ordinates by multiplying it with this 
				# camera's extrinsics matrix
				global_ray = np.dot(self.camera_list[i].extrinsics_absolute, ray)
				# Store the XZ co-ordinates of the global ray separately for easy processing
				global_ray_xz = np.asarray([global_ray[0], global_ray[2]])
				# Find the angle for this ray in the global frame of reference
				theta_ray = mapPointToPanaromaAngle(global_ray_xz, viewing_circle_centre, ipd, eye)
				# Find the normalized column co-ordinate for this ray in the global panaroma
				xn_ray = mapPointToPanaromaColumn( global_ray_xz, viewing_circle_centre, ipd, eye)

				# Unnormalize this column to the width of the panaroma
				panaroma_col_index = int(unnormalizeX(xn_ray, pan_width))
				# Get the image for this camera
				current_image = self.image_list[i]
				# Copy column to the final panaroma
				temp_image[:, panaroma_col_index, :] = current_image.getColumn(col)

			cv2.imshow('Temporary result:', temp_image)
			cv2.waitKey(50)
			# Stack rendering results to average later
			image_stacker[:, :, :, i] = temp_image

		# Final image is some combination of the stack
		output_image = np.uint8(np.max(image_stacker, axis=3))
		return output_image


	def renderCOPSOnly(self, ipd, output_image_dim, eye=-1, origin=[0, 0, 0]):
		self.sanityCheck()
		height = output_image_dim[0]
		width = output_image_dim[1]
		print("ODS width")
		print(width)
		
		output_image = np.zeros((output_image_dim[0], output_image_dim[1], 3), dtype='uint8')

		camera_positions = self.camera_list.getCameraCentresXZ(origin)
		viewing_circle_centre = self.camera_list.getViewingCircleCentre()
		viewing_circle_radius = self.camera_list.getViewingCircleRadius()
		# IPD cannot be greater than the radius of the viewing circle
		if ipd > viewing_circle_radius:
			raise RuntimeError('IPD cannot be greater than the radius of the viewing circle')

		num_cameras = self.camera_list.getNumCameras()
		
		oF =OpticalFlowCalculator()
		
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
				print(col_img)
				image = self.image_list[i]
				output_image[:, col_index, :] = image.getColumn(int(col_img))

		#cameras=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

		#cameras=[0, 1, 2, 3, 8, 9, 6, 7, 4, 5, 0]

		cameras=[0, 5, 4, 7, 6, 9, 8, 3, 2, 1, 0]

		flows=[]
		
		hahacams = 10
		for i in range(hahacams):	

			index0=cameras[i]
			index1=cameras[i+1]
			
			image0=self.image_list[index0].getImage()			
			image1=self.image_list[index1].getImage()
			
			flow_i=oF.calculateFlow(image0, image1)
			flows.append(flow_i)
		
		all_flows=np.asarray(flows)
		print(all_flows.shape)
		
		#view interpolation
		#vertical_pixel=0
		

		#for i in range(num_cameras):

		for i in range(hahacams):

			index0=cameras[i]
			index1=cameras[i+1]
			cam0=self.camera_list[index0]
			cam1=self.camera_list[index1]
			cam_position0=camera_positions[index0, :]
			cam_position1=camera_positions[index1, :]
			
			image0=self.image_list[index0].getImage()			
			image1=self.image_list[index1].getImage()
			
			image_width=int(cam0.resolution[0])
			
			image_height=int(cam0.resolution[1])
			
			relative_theta_0=cam0.getCOPRelativeAngleLeft()
			relative_theta_1=cam1.getCOPRelativeAngleLeft()
			
			#theta_0=normalizedXToTheta(cam0.getPositionInODSImageLeft())
			#theta_1=normalizedXToTheta(cam1.getPositionInODSImageLeft())
			theta_0=mapPointToPanaromaAngle(cam_position0, viewing_circle_centre, ipd, eye)
			theta_1=mapPointToPanaromaAngle(cam_position1, viewing_circle_centre, ipd, eye)
			
			theta_0_degree=radians2Degrees360(theta_0)
			theta_1_degree=radians2Degrees360(theta_1)
			
			print("theta_0_degree")
			print(theta_0_degree)
			print("theta_1_degree")
			print(theta_1_degree)
			
			
			
			x0=int(round(cam0.getCOPLeft()[0]))
			#x0=int(round(cam0.getCOPLeft()))
			print('x0: ', x0)
			print(x0)
			print('Normalized Position x0: ', cam0.getPositionInODSImageLeft())
			
			#col0_flows =all_flows[index0, :,  x0, 1]
			#sum=np.sum(col0_flows)
			#avg=int(sum/image_height)
			#print(avg)
			#x0_flowed=x0+np.abs(avg)
			#relative_theta_0_cam1=getRelativeAngle(cam0.resolution[0], cam_position0, x0_flowed, cam0.favg, cam0.getFieldOfView())
		
			x1=int(round(cam1.getCOPLeft()[0]))
			#x1=int(round(cam1.getCOPLeft()))
			print('x1: ', x1)
			print(x1)
			print( 'Normalized Position x0: ', cam1.getPositionInODSImageLeft())
			#col1_flows =all_flows[index0, :,  x1, 1]
			#sum=np.sum(col1_flows)
			#avg=int(sum/image_height)
			#print(avg)
			#x1_flowed=x1+np.abs(avg)
			#relative_theta_1_cam0=getRelativeAngle(cam1.resolution[0], cam_position1, x1_flowed, cam1.favg, cam1.getFieldOfView())
			
			#print("\n")
			field_of_view = self.camera_list[i].fov_x
			print('FOV Cam: ', radians2Degrees(field_of_view))


			#for j in range(x0+1, image_width):
			for j in range(0, x0):
				#relative_theta_a=getRelativeAngle(cam0.resolution[0], cam_position0, j, cam0.favg, cam0.getFieldOfView())
				ray_a = cam0.getRayForPixel(j, 0)

				ray_a = unit_vector(ray_a)
				ray_a=np.append(ray_a,1)
				global_ray_a=np.dot(cam0.extrinsics_absolute, ray_a)
				global_ray_a_xz=np.asarray([global_ray_a[0], global_ray_a[2]])
				theta_a = mapPointToPanaromaAngle(global_ray_a_xz, viewing_circle_centre, ipd, eye)
				
				#col_flows =all_flows[index0, :,  j, 1]
				col_flows =all_flows[0, :,  j, 1]
				sum=np.sum(col_flows)
				avg=int(sum/image_height)
				# print(avg)
				# j_flowed=j-np.abs(avg)
				j_flowed = j +avg
				
				ray_b = cam1.getRayForPixel(j_flowed, 0)
				ray_b = unit_vector(ray_b)
				ray_b=np.append(ray_b,1)
				global_ray_b=np.dot(cam1.extrinsics_absolute, ray_b)
				global_ray_b_xz=np.asarray([global_ray_b[0], global_ray_b[2]])
				theta_b = mapPointToPanaromaAngle(global_ray_b_xz, viewing_circle_centre, ipd, eye)
				
				# print("theta_a")
				# print(theta_a)
				# print("theta_b")
				# print(theta_b)

				theta_a_degree=radians2Degrees360(theta_a)
				print('ta degrees', theta_a_degree)
				theta_b_degree=radians2Degrees360(theta_b)
				print('tb degrees', theta_b_degree)
				
				#relative_theta_b=getRelativeAngle(cam1.resolution[0], cam_position1, j_flowed, cam1.favg, cam1.getFieldOfView())
				#theta_p=self.weightedAverage(relative_theta_0, relative_theta_1, relative_theta_a, relative_theta_b, theta_0, theta_1)
				#theta_p=self.ourInterpolation(relative_theta_0, relative_theta_1, relative_theta_a, relative_theta_b, theta_0, theta_1)
				#theta_p=self.ourLinearInterpolation(relative_theta_0, relative_theta_1, elative_theta_a, relative_theta_b, theta_0, theta_1)
				
				# theta_p_degree=self.linearInterpolation(theta_0_degree, theta_1_degree, theta_a_degree, theta_b_degree)
				# theta_p_degree = (theta_a_degree + theta_b_degree)/2
				theta_p_degree = self.normalizeThenInterpolate(theta_0_degree, theta_1_degree, theta_a_degree, theta_b_degree)
				print("theta_p")
				print(theta_p_degree)
				
				
				print("\n")
				
				#theta_p=self.myInterpolation(relative_theta_0, relative_theta_1, relative_theta_a, relative_theta_b, theta_0, theta_1, relative_theta_0_cam1, relative_theta_1_cam0)
				
				theta_p=degrees3602Radians(theta_p_degree)

				x_i=thetaToNormalizedX(theta_p)
				#print("x_i: ")
				#print(x_i)
				col_i=int(unnormalizeX(x_i, width))
				image = self.image_list[index0]
				if 0<j<image_width:
					if 0<col_i<width:
						output_image[:, col_i, :] =image.getColumn(j)
					

	

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






