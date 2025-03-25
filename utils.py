import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def normalize(image, mean = None, std = None):
    if mean == None:
       mean = [0.485, 0.456, 0.406]
    if std == None:
       std = [0.229, 0.224, 0.225]
    if image.max() > 1:
        image = image.astype(np.float64)
        image /= 255
    image = (image - mean) / std
    # in addition, roll the axis so that they suit pytorch
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()

def getTrueId(img, model, device = 'cpu'):
    model.to(device)
    img = (torch.from_numpy(img).permute(0,3,1,2)).to(device)
    y_pred = model(img)
    class_ = torch.argmax(y_pred, dim = 1)
    ans = torch.nn.functional.softmax(y_pred,1)
    return class_.item()

def get_sample_data(image_ind, images_path, masks_path, transform):
    try:
        image_path = images_path[image_ind]
        mask_path = masks_path[image_ind]
        image_raw_name = image_path.split("/")[-1].split(".")[0]

        # Load the image
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        transformed_image = transform(image)
        transformed_mask = transform(mask)
        return image_raw_name, transformed_image, transformed_mask
    except PermissionError as e:
        print(f"PermissionError: {e}. Please check the file permissions and try again.")
        return None, None, None
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}. Please check if the file exists at the specified path.")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None, None
    
def get_sample_image(images_path, transform):
    image = Image.open(images_path)
    transformed_image = transform(image)
    return transformed_image


def get_neutral_background(image):
    height, width = image.shape[:2]
    corner_size = int(0.1 * height)  # This will be 22 pixels for your 224x224 image
    top_left = image[:corner_size, :corner_size, :]
    top_right = image[:corner_size, -corner_size:, :]
    bottom_left = image[-corner_size:, :corner_size, :]
    bottom_right = image[-corner_size:, -corner_size:, :]
    average_top_left = np.mean(top_left, axis=(0, 1))
    average_top_right = np.mean(top_right, axis=(0, 1))
    average_bottom_left = np.mean(bottom_left, axis=(0, 1))
    average_bottom_right = np.mean(bottom_right, axis=(0, 1))
    average_all_corners = np.mean([average_top_left, average_top_right, average_bottom_left, average_bottom_right], axis=0)
    average_all_corners_broadcasted = average_all_corners[np.newaxis, np.newaxis, :]
    return average_all_corners_broadcasted  

def get_sorted_indices(val):
    flattened_array = val.flatten()
    sorted_indices = np.argsort(flattened_array)
    return sorted_indices

def get_top_k(val, k, sorted_indices = None):
    """
    Find top k biggest values
    """
    if sorted_indices is None:
      sorted_indices = get_sorted_indices(val)
    top_k_indices = sorted_indices[-k:]
    return top_k_indices

def create_mask_from_indices(val, k, sorted_indices = None):
    """
    Create a mask from the indices of the top k biggest values
    """
    top_k_indices = get_top_k(val, k, sorted_indices)
    mask = np.zeros_like(val, dtype=int)
    top_k_positions = np.unravel_index(top_k_indices, val.shape)
    mask[top_k_positions] = 1

    return mask

def get_raw_important_point(important_val, percentile):
    val = important_val
    threshold = np.percentile(important_val, percentile)
    indexes  = np.where(important_val > threshold)
    second_dim = indexes[0]
    third_dim = indexes[1]
    datapoint = [[second_dim[i],third_dim[i]] for i in range(len(second_dim))]
    datapoint = np.array(datapoint)
    return datapoint

def create_shap_image(shap_value, standard_threshold):
    important_point = get_raw_important_point(shap_value,standard_threshold)
    shap_image = np.zeros(shap_value.shape)
    for point in important_point:
        shap_image[point[0], point[1]] = 1
    return shap_image
 
def plot_saliency_with_topk(ax, saliency, original_image, title, k=500, cmap='hot', img_alpha=0.3, sali_alpha=0.6):

    """
    Creates a plot of saliency map showing only top-k values with the original image overlapped.
    
    Args:
        saliency: Array of saliency/importance values to visualize.
        original_image: The original image to overlay.
        title: Title for the saliency plot.
        k: Number of top values to display.
        cmap: Colormap to use for saliency visualization.
        alpha: Alpha value for the original image overlay (lower = more transparent).
    """
    # Create masked version showing only top-k values
    saliency = saliency.copy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    topk_saliency = np.zeros_like(saliency)
    
    # Get sorted indices
    flat_saliency = saliency.flatten()
    sorted_indices = np.argsort(flat_saliency)[::-1]  # Sort in descending order
    
    # Get top-k indices
    top_k_indices = sorted_indices[:k]
    
    # Create masked version with only top-k values
    flattened_topk = np.zeros_like(saliency.flatten())
    flattened_topk[top_k_indices] = flat_saliency[top_k_indices]
    topk_saliency = flattened_topk.reshape(saliency.shape)
    
    # Create figure with single plot
    # fig, ax = plt.subplots(figsize=(4, 4))
    # fig.patch.set_facecolor('black')
    
    # Set background color
    ax.set_facecolor('white')
    
    # First show the original image with low alpha
    ax.imshow(original_image, cmap='hot', alpha=img_alpha)
    
    # Then overlay the saliency map
    im = ax.imshow(topk_saliency, cmap=cmap, alpha=sali_alpha)
    
    # Set title and remove ticks
    ax.set_title(title, color='white', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add white border
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(2)

    plt.tight_layout()
    return ax

def get_score(model, device, image,img_indice = None):
	image = torch.from_numpy(image).permute(0,3,1,2)
	model.to(device)
	image = image.to(device)
	scores = model(image)
	if img_indice is None:
		img_indice = torch.argmax(scores, dim=1).tolist()
	softmax_scores = torch.nn.functional.softmax(scores, dim=1)
	return softmax_scores[0, img_indice]

def exact_find_d_alpha(model, device, to_explain,val,trueImageInd = None, target_ratio=0.5, neutral_val=0, epsilon=0.005, max_iter=100):
	low, high = 0, val.shape[0]*val.shape[1]
	sorted_indices = get_sorted_indices(val)
	full_score = get_score(model, device, to_explain, trueImageInd)
	# print(f"Full score: {full_score}")
	target_score = full_score * target_ratio
	iter_count = 0
	while high - low > 0 and iter_count < max_iter:
		iter_count += 1
		mid = int((low + high) / 2 )
		mask = create_mask_from_indices(val, mid, sorted_indices)
		mask = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
		background_mask = 1 - mask
		partial_image = to_explain[0]*background_mask + mask*neutral_val
		partial_image = np.expand_dims(partial_image, axis=0).astype(np.float32)
		score = get_score(model, device, partial_image, trueImageInd)
		if abs(score - target_score) < epsilon:
			break
		elif score > target_score:
			low = mid
		else:
			high = mid
	return mid, score 

