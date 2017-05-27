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
		diff_b1=theta_b-theta_1
		diff_0a=theta_0-theta_a
		diff_ba=theta_b-theta_a
		diff_01=theta_0-theta_1
		
		theta_p=((diff_b1*theta_0)+(diff_0a*theta_1))/(diff_ba+diff_01)
		while theta_p<-np.pi:
			theta_p=theta_p+np.pi
			
		theta_p=np.pi+theta_p
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
			
		print("theta_p")
		print(theta_p)
		print("\n")
		
		print("theta_p")
		print(theta_p)
		print("\n")
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
		
		print("factor 1")
		print(factor1)
		print("factor 2")
		print(factor2)
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
			
		print("theta_p")
		print(theta_p)
		print("\n")
		return theta_p
		
	def weightedAverage(self, rtheta_0, rtheta_1, rtheta_a, rtheta_b, theta_0, theta_1):
		f = (rtheta_a/rtheta_0)*theta_0
		s = (1-(rtheta_b/rtheta_1))*theta_1
		theta_p=(f+s)/2
		
		while theta_p<-np.pi:
			theta_p=theta_p+np.pi
			
		theta_p=np.pi+theta_p
		print("theta_p")
		print(theta_p)
		print("\n")
		return theta_p

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
		for i in range(2):
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
		cameras=[0, 1, 2, 3, 8, 9, 6, 7, 4, 5, 0]

		flows=[]
		for i in range(2):	
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
		
		for i in range(0,1):
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
			print("relative theta 0")
			print(relative_theta_0)
			print("relative theta 1")
			print(relative_theta_1)
			print("\n")
			
			theta_0=normalizedXToTheta(cam0.getPositionInODSImageLeft())
			theta_1=normalizedXToTheta(cam1.getPositionInODSImageLeft())
			print("theta_0")
			print(theta_0)
			print("theta_1")
			print(theta_1)
			
			x0=int(round(cam0.getCOPLeft()[0]))
			print("x0: ")
			print(x0)
			print(unnormalizeX(cam0.getPositionInODSImageLeft(), width))
			col0_flows =all_flows[index0, :,  x0, 1]
			sum=np.sum(col0_flows)
			avg=int(sum/image_height)
			#print(avg)
			x0_flowed=x0+np.abs(avg)
			relative_theta_0_cam1=getRelativeAngle(cam0.resolution[0], cam_position0, x0_flowed, cam0.favg, cam0.getFieldOfView())
		
			x1=int(round(cam1.getCOPLeft()[0]))
			print("x1: ")
			print(x1)
			print(unnormalizeX(cam1.getPositionInODSImageLeft(), width))
			col1_flows =all_flows[index0, :,  x1, 1]
			sum=np.sum(col1_flows)
			avg=int(sum/image_height)
			#print(avg)
			x1_flowed=x1+np.abs(avg)
			relative_theta_1_cam0=getRelativeAngle(cam1.resolution[0], cam_position1, x1_flowed, cam1.favg, cam1.getFieldOfView())
			
			#print("\n")
			
			for j in range(x0, image_width):
				relative_theta_a=getRelativeAngle(cam0.resolution[0], cam_position0, j, cam0.favg, cam0.getFieldOfView())
				
				col_flows =all_flows[index0, :,  j, 1]
				sum=np.sum(col_flows)
				avg=int(sum/image_height)
				#print(avg)
				j_flowed=j+np.abs(avg)
				
				relative_theta_b=getRelativeAngle(cam1.resolution[0], cam_position1, j_flowed, cam1.favg, cam1.getFieldOfView())
				
				theta_p=self.weightedAverage(relative_theta_0, relative_theta_1, relative_theta_a, relative_theta_b, theta_0, theta_1)
				#theta_p=self.ourInterpolation(relative_theta_0, relative_theta_1, relative_theta_a, relative_theta_b, theta_0, theta_1)
				#theta_p=self.ourLinearInterpolation(relative_theta_0, relative_theta_1, elative_theta_a, relative_theta_b, theta_0, theta_1)
				#theta_p=self.linearInterpolation(relative_theta_a, relative_theta_b, theta_0, theta_1)
				#theta_p=self.myInterpolation(relative_theta_0, relative_theta_1, relative_theta_a, relative_theta_b, theta_0, theta_1, relative_theta_0_cam1, relative_theta_1_cam0)
				
				x_i=thetaToNormalizedX(theta_p)
				#print("x_i: ")
				#print(x_i)
				col_i=int(unnormalizeX(x_i, width))
				print("col_i: ")
				print(col_i)
				print("\n")
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






