# Imports
import numpy as np
from PIL import Image
import cv2



# Class: Mosaic_Canvas
class Mosaic_Canvas(object):
	def __init__(self, patch_size=256, n=100, downscale=4, n_per_row=10, bg_color=(0,0,0), alpha=-1):
		
		# Class variables
		self.patch_size = patch_size
		self.downscaled_patch_size = int(np.ceil(patch_size/downscale))
		self.n_rows = int(np.ceil(n / n_per_row))
		self.n_cols = n_per_row
		
		# Compute the size of the canvas
		w = self.n_cols * self.downscaled_patch_size
		h = self.n_rows * self.downscaled_patch_size
		if alpha < 0:
			canvas = Image.new(size=(w,h), mode="RGB", color=bg_color)
		else:
			canvas = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))
		
		# Class variables
		self.canvas = canvas
		self.dimensions = np.array([w, h])
		self.reset_coord()

		return


	# Method: Reset coordinates
	def reset_coord(self):
		
		self.coord = np.array([0, 0])

		return


	#  Method: Increment coordinates
	def increment_coord(self):

		# print('current coord: {} x {} / {} x {}'.format(self.coord[0], self.coord[1], self.dimensions[0], self.dimensions[1]))
		assert np.all(self.coord<=self.dimensions)
		
		if self.coord[0] + self.downscaled_patch_size <=self.dimensions[0] - self.downscaled_patch_size:
			self.coord[0]+=self.downscaled_patch_size
		else:
			self.coord[0] = 0 
			self.coord[1]+=self.downscaled_patch_size
		
		return


	# Method: Save coordinates
	def save(self, save_path, **kwargs):
		
		self.canvas.save(save_path, **kwargs)

		return


	# Method: Paste patches
	def paste_patch(self, patch):
		
		assert patch.size[0] == self.patch_size
		assert patch.size[1] == self.patch_size
		
		self.canvas.paste(patch.resize(tuple([self.downscaled_patch_size, self.downscaled_patch_size])), tuple(self.coord))
		self.increment_coord()

		return


	# Method: Get painting (i.e., the canvas)
	def get_painting(self):
		return self.canvas



# Class: Contour_Checking_fn
class Contour_Checking_fn(object):
	
	# Method: __call__
	def __call__(self, pt): 
		raise NotImplementedError



# Class: isInContourV1
class isInContourV1(Contour_Checking_fn):

	# Method: __init__
	def __init__(self, contour):
		self.cont = contour

		return


	# Method: __call__
	def __call__(self, pt): 
		return 1 if cv2.pointPolygonTest(self.cont, tuple(np.array(pt).astype(float)), False) >= 0 else 0



# Method: isInContourV2
class isInContourV2(Contour_Checking_fn):

	# Method: __init__
	def __init__(self, contour, patch_size):

		# Class variables
		self.cont = contour
		self.patch_size = patch_size

		return


	# Method: __call__
	def __call__(self, pt): 
		pt = np.array((pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)).astype(float)
		return 1 if cv2.pointPolygonTest(self.cont, tuple(np.array(pt).astype(float)), False) >= 0 else 0



# Class: isInContourV3_Easy (1 of 4 points need to be in the contour for test to pass)
class isInContourV3_Easy(Contour_Checking_fn):
	
	# Method: __init__
	def __init__(self, contour, patch_size, center_shift=0.5):
		self.cont = contour
		self.patch_size = patch_size
		self.shift = int(patch_size//2*center_shift)

		return


	# Method: __call__
	def __call__(self, pt): 
		center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
		if self.shift > 0:
			all_points = [(center[0]-self.shift, center[1]-self.shift),
						  (center[0]+self.shift, center[1]+self.shift),
						  (center[0]+self.shift, center[1]-self.shift),
						  (center[0]-self.shift, center[1]+self.shift)
						  ]
		else:
			all_points = [center]
		
		for points in all_points:
			if cv2.pointPolygonTest(self.cont, tuple(np.array(points).astype(float)), False) >= 0:
				return 1
		return 0


# Class: isInContourV3_Hard (all 4 points need to be in the contour for test to pass)
class isInContourV3_Hard(Contour_Checking_fn):
	
	# Method: __init__
	def __init__(self, contour, patch_size, center_shift=0.5):
		self.cont = contour
		self.patch_size = patch_size
		self.shift = int(patch_size//2*center_shift)
	
	
	# Method: __call__
	def __call__(self, pt): 
		center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
		if self.shift > 0:
			all_points = [(center[0]-self.shift, center[1]-self.shift),
						  (center[0]+self.shift, center[1]+self.shift),
						  (center[0]+self.shift, center[1]-self.shift),
						  (center[0]-self.shift, center[1]+self.shift)
						  ]
		else:
			all_points = [center]
		
		for points in all_points:
			if cv2.pointPolygonTest(self.cont, tuple(np.array(points).astype(float)), False) < 0:
				return 0
		return 1
