import numpy as np 

from cameras import *
from RayGeometry import *
from viewSynth import *
import matplotlib.patches as mpatches


class RendererODS():
	def __init__(self):
		self.init_complete = False
		self.color_list = ['green', 'blue', 'black', 'orange', 'red', 'cyan',
		'magenta', 'darkgreen', 'purple', 'violet']
		
	def jumpLinearInterpolation(self, theta_0, theta_1,theta_a, theta_b):
		diff_b1=theta_1-theta_b
		diff_0a=theta_0-theta_a
		diff_ba=theta_b-theta_a
		diff_10=theta_0-theta_1
		
		theta_p=((diff_b1*theta_0)+(diff_0a*theta_1))/(diff_ba+diff_10)
		
		return theta_p

	def normalizeThenInterpolate(self, t0, t1, ta, tb, eye=-1):
		if eye==-1:
			# handle special case of the discontinuity between 0 to 360 degrees
			if abs(t0-t1) <= 270.0:
				drange = t0-t1
				t0s = (t0-t0)/drange
				t1s = (t1-t0)/drange
				tas = (ta-t0)/drange
				tbs = (tb-t0)/drange
				tps = self.jumpLinearInterpolation(t0s, t1s, tas, tbs)
				tps = tps*drange + t0
			else:
				t1_new = abs(t1-360)+t0
				tb_new = abs(tb-360)+t0
				drange = (abs(t1_new))*2
				t0s = (t0-t0)/drange
				t1s = (abs(t1_new))/drange
				tas = (t0-ta)/drange
				tbs = (abs(tb_new))/drange
				tps = self.jumpLinearInterpolation(t0s, t1s, tas, tbs)
				tps = t0-(tps*drange)+360
		else:
			# handle special case of the discontinuity between 0 to 360 degrees
			if abs(t0-t1) <= 270.0:
				drange = t0-t1
				t0s = (t0-t0)/drange
				t1s = (t1-t0)/drange
				tas = (ta-t0)/drange
				tbs = (tb-t0)/drange
				tps = self.jumpLinearInterpolation(t0s, t1s, tas, tbs)
				tps = tps*drange + t0
			else:
				t0 = t0+360
				if abs(ta-t1) >= 270:
					ta = ta+360
				if abs(tb-t1) >= 270:
					tb = tb+360
				drange = t0-t1
				t0s = (t0-t0)/drange
				t1s = (t1-t0)/drange
				tas = (ta-t0)/drange
				tbs = (tb-t0)/drange
				tps = self.jumpLinearInterpolation(t0s, t1s, tas, tbs)
				tps = tps*drange + t0

		return tps

	def pixRGB2Gray(self, pix):
		assert(pix.shape[0]==3)
		return 0.2989 * pix[0] + 0.5870 * pix[1] + 0.1140 * pix[2]

	def getBrighterPixel(self, pix1, pix2):
		if self.pixRGB2Gray(pix1) >= self.pixRGB2Gray(pix2):
			return pix1
		else:
			return pix2


	def updateODSPanaroma(self, panaroma, temp_result):
		num_cols = temp_result.shape[1]
		for i in range(num_cols):
			col_vals = temp_result[:, i, :]
			sum_col_vals = np.sum(col_vals)
			if sum_col_vals == 0:
				panaroma[:, i, :] = panaroma[:, i, :] + temp_result[:, i, :]
			else:
				pan_sum = np.sum(panaroma[:, i, :])
				if pan_sum == 0:
					panaroma[:, i, :] = temp_result[:, i, :]
				else:
					panaroma[:, i, :] = 0.5*panaroma[:, i, :] + 0.5*temp_result[:, i, :]
		return panaroma


	def setCameraList(self, camera_collection):
		self.camera_list = camera_collection
		self.camera_order = range(self.camera_list.getNumCameras())
		self.camera_order.append(0)
		self.init_complete = True

	def setCameraOrder(self, camera_order):
		self.camera_order = camera_order

	def setImageList(self, image_collection):
		self.image_list = image_collection

	def sanityCheck(self):
		if not self.init_complete:
			raise RuntimeError('Camera collection is not initialized')

	def rigVisTest(self, ipd, out_image_dim, eye=-1, origin=[0, 0, 0]):
			self.sanityCheck()
			height = out_image_dim[0]
			width = out_image_dim[1]
			output_image = np.zeros((out_image_dim[0], out_image_dim[1], 3), dtype='uint8')

			camera_positions = self.camera_list.getCameraCentresXZ(origin)
			viewing_circle_centre = self.camera_list.getViewingCircleCentre()
			camera_rig_radius = self.camera_list.getViewingCircleRadius()

			if ipd > camera_rig_radius:
				raise RuntimeError('IPD cannot be greater than the radius of the viewing circle')

			nc = self.camera_list.getNumCameras()
			# Plots, plots and more plots
			fig, ax = pyplt.subplots()
			ax.hold('on')
			for i in range(0, nc):
				theta = getAngle(viewing_circle_centre, camera_positions[i, :], ipd)
				self.camera_list[i].setCOPRelativeAngleRight(theta)
				self.camera_list[i].setCOPRelativeAngleLeft(theta)
				xn_right = mapPointToODSColumn(camera_positions[i, :], viewing_circle_centre, ipd, 1)
				xn_left = mapPointToODSColumn(camera_positions[i, :], viewing_circle_centre, ipd, -1)
				self.camera_list[i].setPositionInODSImageRight(xn_right)
				self.camera_list[i].setPositionInODSImageLeft(xn_left)

				
				# Plot origin and the viewing circle centre.
				ax.scatter(0, 0)
				ax.annotate('O', (0, 0))
				ax.annotate('VC', (viewing_circle_centre[0], viewing_circle_centre[1]))
				ax.scatter(viewing_circle_centre[0], viewing_circle_centre[1], color='blue')

				# Plot cameras in XZ.
				ax.annotate('C', (camera_positions[i, 0], camera_positions[i, 1]), color='green')
				ax.scatter(camera_positions[i, 0], camera_positions[i, 1], color='blue')

				# Plot viewing circle
				viewing_circle = getCirclePoints(viewing_circle_centre, ipd/2)
				ax.scatter(viewing_circle[:, 0],viewing_circle[:, 1], color='red')

			legend_labels = []
			for i in range(0, 10):
				image_width = int(self.camera_list[i].resolution[0])
				if eye is 1:
					xn = self.camera_list[i].getPositionInODSImageRight()
					col_img = self.camera_list[i].getCOPRight()
				else:
					xn = self.camera_list[i].getPositionInODSImageLeft()
					col_img = self.camera_list[i].getCOPLeft()
				col_index = int(unnormalizeX(xn, width))
				print(col_img, image_width)
				for j in range(0, image_width):
					point_in_3d = self.camera_list[i].getRayForPixelInGlobalRef(j, 0)
					p3d = np.asarray((point_in_3d[0], point_in_3d[2]))
					point = get2DPointOnODSVC(p3d, viewing_circle_centre, ipd, eye)
					# Find where these rays map onto the ODS panaroma
					xn_new = mapPointToODSColumn(p3d, viewing_circle_centre, ipd, eye)
					# print(xn, xn_new)
					# Plot where the start, middle and end of an image are mapped in the global frame of reference
					ax.scatter(point_in_3d[0], point_in_3d[2], color=self.color_list[i])
					ax.scatter(point[0], point[1], color=self.color_list[i])
				# Set legend labels
				legend_labels.append(mpatches.Patch(color=self.color_list[i], label='cam ' + str(i)))
			pyplt.legend(handles=legend_labels)
			pyplt.show()


	def render360NoInterpolation(self, ipd, out_image_dim, eye=1, origin=[0, 0, 0]):
		self.sanityCheck()
		height = out_image_dim[0]
		pan_width = out_image_dim[1]
		output_image = np.zeros((out_image_dim[0], out_image_dim[1], 3), dtype='uint8')
		nc = self.camera_list.getNumCameras()
		image_stacker = np.zeros((out_image_dim[0], out_image_dim[1], 3, nc), dtype='uint8')

		camera_positions = self.camera_list.getCameraCentresXZ(origin)
		viewing_circle_centre = self.camera_list.getViewingCircleCentre()
		rig_radius = self.camera_list.getViewingCircleRadius()

		if ipd > rig_radius:
			raise RuntimeError('IPD cannot be greater than the radius of the camera rig.')

		for i in range(0, nc):
			theta = getAngle(viewing_circle_centre, camera_positions[i, :], ipd)
			self.camera_list[i].setCOPRelativeAngleRight(theta)
			self.camera_list[i].setCOPRelativeAngleLeft(theta)
			
			xn_right = mapPointToODSColumn(camera_positions[i, :], viewing_circle_centre, ipd, eye=1)
			xn_left = mapPointToODSColumn(camera_positions[i, :], viewing_circle_centre, ipd, eye=-1)
			self.camera_list[i].setPositionInODSImageRight(xn_right)
			self.camera_list[i].setPositionInODSImageLeft(xn_left)

			if eye is 1:
				xn = xn_right
			else:
				xn = xn_left

			col_index = int(unnormalizeX(xn, pan_width))
			# Map this camera to an angle on the viewing circle.
			cam_theta = mapPointToODSAngle(camera_positions[i, :], viewing_circle_centre, ipd, eye)

			print('camera ', i, 'maps to: ', 'angle: ', radians2Degrees(cam_theta), 'column: ',  col_index)
			
			print('Rendering camera ', i)
			# Temporary image that holds the render result just for this camera
			temp_image = np.zeros((out_image_dim[0], out_image_dim[1], 3), dtype='uint8')
			# Go over all columns in the image and get a ray for each column.
			for col in range(0, int(self.camera_list[i].resolution[0])):
				# Setting row index to zero, because it doesn't really matter
				global_ray = self.camera_list[i].getRayForPixelInGlobalRef(col, 0)
				# Store the XZ co-ordinates of the global ray separately for easy processing
				global_ray_xz = np.asarray([global_ray[0], global_ray[2]])
				# Find the angle for this ray in the global frame of reference
				theta_ray = mapPointToODSAngle(global_ray_xz, viewing_circle_centre, ipd, eye)
				# Find the normalized column co-ordinate for this ray in the global panaroma
				xn_ray = mapPointToODSColumn( global_ray_xz, viewing_circle_centre, ipd, eye)

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


	# View interpolater - One flow vector for an entire column
	def viewInterpolationCwise(self, cameraLeftID, cameraRightID, frameIDLeft, frameIDRight, pan_width, 
		direction='left2right', origin=[0, 0, 0], ipd=0.062, eye=-1):
		# Do sanity checks
		# Check if all IDs are valid.
		camLeft = self.camera_list[cameraLeftID]
		camRight = self.camera_list[cameraRightID]
		imageLeft = self.image_list[frameIDLeft].getImage()
		imageRight = self.image_list[frameIDRight].getImage()

		# Resolution of both the left and right are assumed to be the same. Taking the left image
		# here as the reference.
		image_width = camLeft.resolution[0]
		image_height = camLeft.resolution[1]
		output_image = np.zeros((int(image_height), int(pan_width), 3), dtype='uint8')

		# get and store renderer globals locally for computation
		camera_positions = self.camera_list.getCameraCentresXZ(origin)
		viewing_circle_centre = self.camera_list.getViewingCircleCentre()
		camera_rig_radius = self.camera_list.getViewingCircleRadius()

		cameraLeftPosition = camera_positions[cameraLeftID]
		cameraRightPosition = camera_positions[cameraRightID]

		# Find where the two incoming cameras map onto the viewing circle. These form theta_0 and theta_1
		theta_0 = mapPointToODSAngle(cameraLeftPosition, viewing_circle_centre, ipd, eye)
		theta_1 = mapPointToODSAngle(cameraRightPosition, viewing_circle_centre, ipd, eye)
		theta_0_degree = radians2Degrees360(theta_0)
		theta_1_degree = radians2Degrees360(theta_1)

		if direction is 'left2right':
			start_col = camLeft.getCOPLeft()
			end_col = image_width
			camFirst = camLeft
			camSecond = camRight
			imageFirst = imageLeft
			imageSecond = imageRight
		elif direction is 'right2left':
			start_col = 0
			end_col = camLeft.getCOPLeft()
			camFirst = camRight
			camSecond = camLeft
			imageFirst = imageRight
			imageSecond = imageLeft
		else:
			raise RuntimeError('Unsupported view interpolation direction : ', direction)

		# Debug prints
		# print('start_col: ', start_col, '\tend col: ', end_col)
		# print('theta_0: ', theta_0_degree, '\ttheta_1: ', theta_1_degree)

		# Optical flow calculator
		of = OpticalFlowCalculator()
		flow = of.calculateFlow(imageFirst, imageSecond)

		for col_id in range(start_col, end_col):
			ray_first = camFirst.getRayForPixelInGlobalRef(col_id, 0)
			ray_first_xz = np.asarray((ray_first[0], ray_first[2]), dtype='float32')
			theta_a = mapPointToODSAngle(ray_first_xz, viewing_circle_centre, ipd, eye)
			theta_a_degree = radians2Degrees360(theta_a)

			# Get horizontal flow for the current column
			col_flow = flow[:, col_id, 1]
			mean_flow = np.mean(col_flow)

			col_id_correspondence = col_id + mean_flow
			ray_second = camSecond.getRayForPixelInGlobalRef(col_id_correspondence, 0)
			ray_second_xz = np.asarray((ray_second[0], ray_second[2]), dtype='float32')
			theta_b = mapPointToODSAngle(ray_second_xz, viewing_circle_centre, ipd, eye)
			theta_b_degree = radians2Degrees360(theta_b)

			theta_p_degree = self.normalizeThenInterpolate(theta_0_degree, theta_1_degree, theta_a_degree, theta_b_degree, eye)
			theta_p = degrees3602Radians(theta_p_degree)

			xn_new = thetaToNormalizedX(theta_p)
			# Make sure xn_new is between 0 and 1.
			xn_new = np.clip(xn_new, 0, 1)
			ods_column = int(unnormalizeX(xn_new, pan_width))
			# print(col_id, theta_a_degree, theta_b_degree, theta_p_degree, ods_column)
			output_image[:, ods_column, :] = imageFirst[:, col_id, :]

		return output_image

	# View interpoaltion: Per pixel flow
	def viewInterpolationPixelwise(self, cameraLeftID, cameraRightID, frameIDLeft, frameIDRight, pan_width, 
		direction='left2right', origin=[0, 0, 0], ipd=0.062, eye=-1):
		# Do sanity checks
		# Check if all IDs are valid.
		camLeft = self.camera_list[cameraLeftID]
		camRight = self.camera_list[cameraRightID]
		imageLeft = self.image_list[frameIDLeft].getImage()
		imageRight = self.image_list[frameIDRight].getImage()

		# Resolution of both the left and right are assumed to be the same. Taking the left image
		# here as the reference.
		image_width = camLeft.resolution[0]
		image_height = camLeft.resolution[1]
		output_image = np.zeros((int(image_height), int(pan_width), 3), dtype='uint8')

		# get and store renderer globals locally for computation
		camera_positions = self.camera_list.getCameraCentresXZ(origin)
		viewing_circle_centre = self.camera_list.getViewingCircleCentre()
		camera_rig_radius = self.camera_list.getViewingCircleRadius()

		cameraLeftPosition = camera_positions[cameraLeftID]
		cameraRightPosition = camera_positions[cameraRightID]

		# Find where the two incoming cameras map onto the viewing circle. These form theta_0 and theta_1
		theta_0 = mapPointToODSAngle(cameraLeftPosition, viewing_circle_centre, ipd, eye)
		theta_1 = mapPointToODSAngle(cameraRightPosition, viewing_circle_centre, ipd, eye)
		theta_0_degree = radians2Degrees360(theta_0)
		theta_1_degree = radians2Degrees360(theta_1)

		if direction is 'left2right':
			start_col = camLeft.getCOPLeft()
			end_col = image_width
			camFirst = camLeft
			camSecond = camRight
			imageFirst = imageLeft
			imageSecond = imageRight
		elif direction is 'right2left':
			start_col = 0
			end_col = camLeft.getCOPLeft()
			camFirst = camRight
			camSecond = camLeft
			imageFirst = imageRight
			imageSecond = imageLeft
		else:
			raise RuntimeError('Unsupported view interpolation direction : ', direction)

		# Debug prints
		# print('start_col: ', start_col, '\tend col: ', end_col)
		# print('theta_0: ', theta_0_degree, '\ttheta_1: ', theta_1_degree)

		# Optical flow calculator
		of = OpticalFlowCalculator()
		flow = of.calculateFlow(imageFirst, imageSecond)

		for col_id in range(start_col, end_col):
			for row_id in range(0, image_height):
				ray_first = camFirst.getRayForPixelInGlobalRef(col_id, row_id)
				ray_first_xz = np.asarray((ray_first[0], ray_first[2]), dtype='float32')
				theta_a = mapPointToODSAngle(ray_first_xz, viewing_circle_centre, ipd, eye)
				theta_a_degree = radians2Degrees360(theta_a)

				# Get horizontal flow for the current column
				ver_flow = flow[row_id, col_id, 0]
				hor_flow = flow[row_id, col_id, 1]
				col_id_correspondence = col_id + hor_flow
				row_id_correspondence = row_id + ver_flow
				ray_second = camSecond.getRayForPixelInGlobalRef(col_id_correspondence, row_id_correspondence)
				ray_second_xz = np.asarray((ray_second[0], ray_second[2]), dtype='float32')
				theta_b = mapPointToODSAngle(ray_second_xz, viewing_circle_centre, ipd, eye)
				theta_b_degree = radians2Degrees360(theta_b)

				theta_p_degree = self.normalizeThenInterpolate(theta_0_degree, theta_1_degree, theta_a_degree, theta_b_degree, eye)
				theta_p = degrees3602Radians(theta_p_degree)

				xn_new = thetaToNormalizedX(theta_p)
				# Make sure xn_new is between 0 and 1.
				xn_new = np.clip(xn_new, 0, 1)
				ods_column = int(unnormalizeX(xn_new, pan_width))
				# print(col_id, theta_a_degree, theta_b_degree, theta_p_degree, ods_column)
				output_image[row_id, ods_column, :] = imageFirst[row_id, col_id, :]

		return output_image

	# View interpolation wrapper : default is column wise interpolation
	def viewInterpolate(self, cameraLeftID, cameraRightID, frameIDLeft, frameIDRight, pan_width, 
		direction='left2right', origin=[0, 0, 0], ipd=0.062, eye=-1, vi_type='cwise'):
		if vi_type is 'cwise':
			interp_image = self.viewInterpolationCwise(cameraLeftID, cameraRightID, frameIDLeft, frameIDRight,
				pan_width, direction, origin, ipd, eye)
		elif vi_type is 'pwise':
			interp_image = self.viewInterpolationPixelwise(cameraLeftID, cameraRightID, frameIDLeft, frameIDRight,
				pan_width, direction, origin, ipd, eye)
		else:
			raise RuntimeError('Unknown interpolation type')

		return interp_image


	def render360WithViewInterpolation(self, ipd, output_image_dim, eye=-1, origin=[0, 0, 0]):
		self.sanityCheck()
		height = output_image_dim[0]
		width = output_image_dim[1]
		output_image = np.zeros((height, width, 3), dtype='uint8')

		camera_positions = self.camera_list.getCameraCentresXZ(origin)
		viewing_circle_centre = self.camera_list.getViewingCircleCentre()
		rig_radius = self.camera_list.getViewingCircleRadius()

		if ipd > rig_radius:
			raise RuntimeError('IPD too large')

		nc = self.camera_list.getNumCameras()
		# Order of the cameras in the rig
		camera_order = self.camera_order
		# setup cameras for rendering
		for i in range(0, nc):
			theta = getAngle(viewing_circle_centre, camera_positions[i, :], ipd)
			self.camera_list[i].setCOPRelativeAngleLeft(theta)
			self.camera_list[i].setCOPRelativeAngleRight(theta)

			xnl = mapPointToODSColumn(camera_positions[i, :], viewing_circle_centre, ipd, -1)
			xnr = mapPointToODSColumn(camera_positions[i, :], viewing_circle_centre, ipd, 1)
			self.camera_list[i].setPositionInODSImageLeft(xnl)
			self.camera_list[i].setPositionInODSImageRight(xnr)

		# View interpolation
		cam_start = 0
		cam_end = 10
		for cam in range(cam_start, cam_end, 2):
			tempL2R = self.viewInterpolate(camera_order[cam], camera_order[cam+1], 
				camera_order[cam], camera_order[cam+1], width, direction='left2right',
				origin=origin, ipd=ipd, eye=eye, vi_type='cwise')
			output_image = self.updateODSPanaroma(output_image, tempL2R)

		for cam in range(cam_start, cam_end, 2):
			tempR2L = self.viewInterpolate(camera_order[cam], camera_order[cam+1], 
				camera_order[cam], camera_order[cam+1], width, direction='right2left',
				origin=origin, ipd=ipd, eye=eye, vi_type='cwise')
			output_image = self.updateODSPanaroma(output_image, tempR2L)

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






