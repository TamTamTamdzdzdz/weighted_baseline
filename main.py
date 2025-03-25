import argparse
import os

from torchvision import models
from torchvision import transforms

from explainer import MyExplainer
from utils import *

NSAMPLE = 3
RSEED = 6
torch.manual_seed(RSEED)
np.random.seed(RSEED)

def explain_img(img_path, transform, model, device, num_points=500, img_alpha=0.3, sali_alpha=0.6):
	transformed_image = get_sample_image(img_path, transform)
	to_explain = np.array(transformed_image.permute(1,2,0).unsqueeze(0))

	white_baseline = np.ones(to_explain.shape)
	black_baseline = np.zeros(to_explain.shape)
	random_baseline = np.random.rand(*to_explain.shape)
	baseline = np.concatenate([white_baseline, black_baseline, random_baseline], axis = 0)
	trueImageInd = getTrueId(to_explain, model, device)
	average_all_corners_broadcasted = get_neutral_background(to_explain[0])
	normalized_baseline = normalize(baseline).to(device)
	explainer = MyExplainer(model.to(device), normalized_baseline, local_smoothing = 0)

	shap_values, indexes, baseline_samples, individual_grads = explainer.shap_values(
			normalize(to_explain).to(device), ranked_outputs=1, nsamples = NSAMPLE, rseed = RSEED)
	shap_values = [np.swapaxes(s, 0, -1) for s in shap_values]
	raw_shap_value = np.sum(shap_values[0], axis = (0,-1))
	weight_list = []
	for ind in range(len(individual_grads)):
		individual_grad = np.abs(individual_grads[ind])
		individual_val = np.sum(individual_grad, axis = 0)
		num_of_deleted_point, score = exact_find_d_alpha(model, device, to_explain,individual_val,trueImageInd = trueImageInd, 
														 target_ratio=0.5, neutral_val=average_all_corners_broadcasted, epsilon=0.005, max_iter=100)
		weight_list.append(num_of_deleted_point)   
	weight_list = np.array(weight_list)
	weight_list = (50176/weight_list)/(50176/weight_list).sum()
	weight_list = weight_list.reshape(-1,1,1,1)
	weighted_basedline_shap_val = np.sum(weight_list * individual_grads, axis = (0,1))
	fig, ax = plt.subplots(figsize=(4, 4))
	ax = plot_saliency_with_topk(ax, weighted_basedline_shap_val, to_explain.squeeze().copy(), "Weighted Baseline SHAP", k=num_points, img_alpha=img_alpha, sali_alpha=sali_alpha)
	return fig, ax

def main():
	# Create argument parser
	parser = argparse.ArgumentParser(description='Generate saliency map for an image')
	parser.add_argument('--input', type=str, required=True, help='Path to input image')
	parser.add_argument('--output', type=str, required=True, help='Output directory for saving results')
	args = parser.parse_args()

	os.makedirs(args.output, exist_ok=True)

	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor()
	])

	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

	# Get filename without extension for saving
	img_basename = os.path.splitext(os.path.basename(args.input))[0]
	output_path = os.path.join(args.output, f"{img_basename}_explained.png")

	print(f"Processing image: {args.input}")

	# Generate and save the saliency map
	fig, ax = explain_img(args.input, transform, model, DEVICE, num_points=1000, img_alpha=0.75, sali_alpha=0.8)

	print(f"Saving result to: {output_path}")
	fig.savefig(output_path)

	print("Done!")

if __name__ == '__main__':
	main()



