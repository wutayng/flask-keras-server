import numpy as np
from PIL import Image

def heatmaps(predictions, cmap, vmin):
	"""
	Convert Model Prediction Array to Heatmaps

	args:
		predictions - prediction tensor, shape=(n x res x res x num_joints)
		cmap - matplotlib colormap for heatmap representation
		vmin - heatmap transparency lower threshold

	return:
		heatmap_tensor - 5d tensor of heatmap images, shape=(n x res x res x RGBA x num_joints)
	"""
	tensor_list = []
	for sample in range(predictions.shape[0]):
		img_heatmaps = []
		for joint in range(predictions.shape[3]):
			# Convert Predction Array to Matplotlib Colormap RGB
			colored_image = cmap(predictions[0, :, :, joint])
			# Set Transparent Values below Threshold
			colored_image[:, :, 3][predictions[0, :, :, joint] < vmin] = 0
			# Covert to Array and Append Heatmap List
			colored_array = (colored_image[:, :, :] * 255).astype(np.uint8)
			img_heatmaps.append(colored_array)

		tensor_list.append(np.stack(img_heatmaps, axis=3)) 

	return np.stack(tensor_list, axis=0)


def draw_heatmaps(image, heatmaps):
	"""
	Draw Heatmaps on Top of Original Image

	args:
		image - original image (PIL RGB Image format)
		heatmaps - heatmaps UNSCALED 4d array, shape=(res x res x RGBA x num_joints)

	return:
		image - output image w/ heatmaps (PIL RGB Image format)
	"""
	image.putalpha(255)
	for joint in range(heatmaps.shape[3]):
		# Convert Heatmap Array to Img
		heatmap = Image.fromarray(heatmaps[:, :, :, joint])

		# Resize to Original (Max) Image Dimensions to Paste
		max_dim = max(image.size[0],image.size[1])
		heatmap_resized = heatmap.resize((max_dim, max_dim))

		# Paste Predictions on Original Image (Special Paste to Preserve Alpha)
		image.paste(heatmap_resized, (0, 0), heatmap_resized)

	return image.convert("RGB")



