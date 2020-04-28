import numpy as np
from PIL import Image

def shape(image, resolution):
	"""
	Reshape Any Image to Desired Resolution
	Add Black Padding on right/bottom to preserve AR

	args:
		image - image array to be transformed
		resolution - resolution of output image

	return:
		output_image - resized/padded image array
	"""
	[orig_x, orig_y, channels] = image.shape
	max_dim = max(orig_x,orig_y)
	im_square = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
	im_square.paste(Image.fromarray(image), box=None)
	output_image = im_square.resize((resolution, resolution))

	return np.asarray(output_image)