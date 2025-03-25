import torch
import numpy as np
from PIL import Image
import warnings
from packaging import version
import torch.nn as nn

class MyExplainer():
	def __init__(self, model, data, batch_size=50, local_smoothing=0):
		

		if version.parse(torch.__version__) < version.parse("0.4"):
			warnings.warn("Your PyTorch version is older than 0.4 and not supported.")

		# check if we have multiple inputs
		self.multi_input = False
		if isinstance(data, list):
			self.multi_input = True
		if not isinstance(data, list):
			data = [data]

		# for consistency, the method signature calls for data as the model input.
		# However, within this class, self.model_inputs is the input (i.e. the data passed by the user)
		# and self.data is the background data for the layer we want to assign importances to. If this layer is
		# the input, then self.data = self.model_inputs
		self.model_inputs = data
		self.batch_size = batch_size
		self.local_smoothing = local_smoothing

		self.layer = None
		self.input_handle = None
		self.interim = False
		if isinstance(model, tuple):
			self.interim = True
			model, layer = model
			model = model.eval()
			self.add_handles(layer)
			self.layer = layer

			# now, if we are taking an interim layer, the 'data' is going to be the input
			# of the interim layer; we will capture this using a forward hook
			with torch.no_grad():
				_ = model(*data)
				interim_inputs = self.layer.target_input
				if type(interim_inputs) is tuple:
					# this should always be true, but just to be safe
					self.data = [i.clone().detach() for i in interim_inputs]
				else:
					self.data = [interim_inputs.clone().detach()]
		else:
			self.data = data
		self.model = model.eval()

		multi_output = False
		outputs = self.model(*self.model_inputs)
		if len(outputs.shape) > 1 and outputs.shape[1] > 1:
			multi_output = True
		self.multi_output = multi_output

		if not self.multi_output:
			self.gradients = [None]
		else:
			self.gradients = [None for _ in range(outputs.shape[1])]

	def gradient(self, idx, inputs):
		import torch

		self.model.zero_grad()
		X = [x.requires_grad_() for x in inputs]
		outputs = self.model(*X)
		selected = [val for val in outputs[:, idx]]
		if self.input_handle is not None:
			interim_inputs = self.layer.target_input
			grads = [
				torch.autograd.grad(selected, input, retain_graph=True if idx + 1 < len(interim_inputs) else None)[0]
				.cpu()
				.numpy()
				for idx, input in enumerate(interim_inputs)
			]
			del self.layer.target_input
		else:
			grads = [
				torch.autograd.grad(selected, x, retain_graph=True if idx + 1 < len(X) else None)[0].cpu().numpy()
				for idx, x in enumerate(X)
			]
		return grads

	@staticmethod
	def get_interim_input(self, input, output):
		try:
			del self.target_input
		except AttributeError:
			pass
		self.target_input = input

	def add_handles(self, layer):
		input_handle = layer.register_forward_hook(self.get_interim_input)
		self.input_handle = input_handle

	def shap_values(
		self, X, nsamples=200, ranked_outputs=None, output_rank_order="max", rseed=None, return_variances=False
	):

		# check if we have multiple inputs
		if not self.multi_input:
			assert not isinstance(X, list), "Expected a single tensor model input!"
			X = [X]
		else:
			assert isinstance(X, list), "Expected a list of model inputs!"

		if ranked_outputs is not None and self.multi_output:
			with torch.no_grad():
				model_output_values = self.model(*X)
			# rank and determine the model outputs that we will explain
			if output_rank_order == "max":
				_, model_output_ranks = torch.sort(model_output_values, descending=True)
			elif output_rank_order == "min":
				_, model_output_ranks = torch.sort(model_output_values, descending=False)
			elif output_rank_order == "max_abs":
				_, model_output_ranks = torch.sort(torch.abs(model_output_values), descending=True)
			else:
				emsg = "output_rank_order must be max, min, or max_abs!"
				raise ValueError(emsg)
			model_output_ranks = model_output_ranks[:, :ranked_outputs]
		else:
			model_output_ranks = (
				torch.ones((X[0].shape[0], len(self.gradients))).int() * torch.arange(0, len(self.gradients)).int()
			)

		# if a cleanup happened, we need to add the handles back
		# this allows shap_values to be called multiple times, but the model to be
		# 'clean' at the end of each run for other uses
		if self.input_handle is None and self.interim is True:
			self.add_handles(self.layer)

		# compute the attributions
		X_batches = X[0].shape[0]
		output_phis = []
		output_phi_vars = []
		# samples_input = input to the model
		# samples_delta = (x - x') for the input being explained - may be an interim input
		samples_input = [torch.zeros((nsamples,) + X[t].shape[1:], device=X[t].device) for t in range(len(X))]
		samples_delta = [np.zeros((nsamples,) + self.data[t].shape[1:]) for t in range(len(self.data))]
		baseline_samples = []
		if rseed is None:
			rseed = np.random.randint(0, 1e6)
		for i in range(model_output_ranks.shape[1]):
			np.random.seed(rseed)  # so we get the same noise patterns for each output class
			phis = []
			phi_vars = []
			for k in range(len(self.data)): #self.data = list each component has shape of (1,3,224,224)
				# for each of the inputs being explained - may be an interim input
				phis.append(np.zeros((X_batches,) + self.data[k].shape[1:]))
				phi_vars.append(np.zeros((X_batches,) + self.data[k].shape[1:]))
			for j in range(X[0].shape[0]): #X[0].shape = [1,3,224,224]  X là ảnh cần diễn giải, self.data = baseline
				# fill in the samples arrays
				rind = -1
				for k in range(nsamples):
					# rind = np.random.choice(self.data[0].shape[0]) #Choose a random baseline in set of baselines
					if rind < self.data[0].shape[0]-1:
					  rind = rind + 1
					else:
					  rind = np.random.choice(self.data[0].shape[0]) #Choose a random baseline in set of baselines
					t = np.random.uniform()
					for a in range(len(X)):
						if self.local_smoothing > 0:
							# local smoothing is added to the base input, unlike in the TF gradient explainer
							x = (
								X[a][j].clone().detach()
								+ torch.empty(X[a][j].shape, device=X[a].device).normal_() * self.local_smoothing
							)
						else:
							x = X[a][j].clone().detach()
						samples_input[a][k] = (
							(t * x + (1 - t) * (self.model_inputs[a][rind]).clone().detach()).clone().detach() #Get interpolation point
						)
						baseline_samples.append(x)
						if self.input_handle is None:
							samples_delta[a][k] = (x - (self.data[a][rind]).clone().detach()).cpu().numpy()

					if self.interim is True:
						with torch.no_grad():
							_ = self.model(*[samples_input[a][k].unsqueeze(0) for a in range(len(X))])
							interim_inputs = self.layer.target_input
							del self.layer.target_input
							if type(interim_inputs) is tuple:
								if type(interim_inputs) is tuple:
									# this should always be true, but just to be safe
									for a in range(len(interim_inputs)):
										samples_delta[a][k] = interim_inputs[a].cpu().numpy()
								else:
									samples_delta[0][k] = interim_inputs.cpu().numpy()

				# compute the gradients at all the sample points
				find = model_output_ranks[j, i]
				grads = []
				for b in range(0, nsamples, self.batch_size):
					batch = [
						samples_input[c][b : min(b + self.batch_size, nsamples)].clone().detach() for c in range(len(X))
					]
					grads.append(self.gradient(find, batch))
				grad = [np.concatenate([g[z] for g in grads], 0) for z in range(len(self.data))]
				# assign the attributions to the right part of the output arrays
				for t in range(len(self.data)):
					samples = grad[t] * samples_delta[t]
					phis[t][j] = samples.mean(0)
					phi_vars[t][j] = samples.var(0) / np.sqrt(samples.shape[0])  # estimate variance of means

			output_phis.append(phis[0] if len(self.data) == 1 else phis)
			output_phi_vars.append(phi_vars[0] if not self.multi_input else phi_vars)
		if isinstance(output_phis, list):
			# in this case we have multiple inputs and potentially multiple outputs
			if isinstance(output_phis[0], list):
				output_phis = [np.stack([phi[i] for phi in output_phis], axis=-1) for i in range(len(output_phis[0]))]
			# multiple outputs case
			else:
				output_phis = np.stack(output_phis, axis=-1)

		if ranked_outputs is not None:
			if return_variances:
				return output_phis, output_phi_vars, model_output_ranks
			else:
				return output_phis, model_output_ranks, baseline_samples, samples
		else:
			if return_variances:
				return output_phis, output_phi_vars
			else:
				return output_phis
