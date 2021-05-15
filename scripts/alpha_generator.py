from abc import abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import pdb
import torch
import torch.nn.functional as F
import ast


def add_weighter_args(parser):
	parser.add_argument(
							'-weight-strgy', type=str, default='default',
							choices=['default', 'alt', 'warm_up_down', 'phase_in', 'meta']
	)
	# add arguments for alternating
	parser.add_argument('-alt-freq', type=int, default=1, help='If using alt strategy, how often to alternate')
	parser.add_argument('-prim-start', type=int, default=0, help='What epoch to start training on the primary loss')
	parser.add_argument('-init-val', type=float, default=1.0, help='Initial Task weightings')
	parser.add_argument('-end-val', type=float, default=1.0, help='Final task weightings')


def get_alpha_generator(opts, prim_key, aux_keys):
	weight_strgy = opts.weight_strgy
	if weight_strgy == 'default':
		if not hasattr(opts, 'default_weight_config'):
			setattr(opts, 'default_weight_config', None)
		return DefaultWeighter(prim_key, aux_keys, init_val=opts.init_val, load_config=opts.default_weight_config)
	elif weight_strgy == 'alt':
		return AlternatingWeighter(
					prim_key, aux_keys, init_val=opts.init_val,
					alt_freq=opts.alt_freq, prim_start=opts.prim_start
				)
	elif weight_strgy == 'warm_up_down':
		return WarmUpAndDown(
					prim_key, aux_keys, init_val=opts.init_val,
					prim_first=(opts.prim_start == 0), end_val=opts.end_val,
					reset_every=opts.alt_freq
				)
	elif weight_strgy == 'phase_in':
		return PhaseIn(
					prim_key, aux_keys, init_val=opts.init_val,
					prim_start=opts.prim_start, max_epochs=opts.train_epochs
				)
	elif weight_strgy == 'meta':
		return MetaWeighter(prim_key, aux_keys, meta_lr_weight=opts.meta_lr_weight)
	else:
		assert 'Invalid Value for Weighting strategy - {}'.format(weight_strgy)


class Weighter(object):
	def __init__(self, prim_key, aux_keys, init_val=1.0):
		self.weights = {key: init_val for key in aux_keys}
		self.aux_keys = aux_keys
		self.weights[prim_key] = init_val
		self.prim_key = prim_key
		self.init_val = init_val
		self.result_logs = []
		self.class_norm_logs = []
		self.is_meta = False

	def __getitem__(self, key):
		return self.weights[key]

	@abstractmethod
	def prep_epoch_start(self, epoch, **kwargs):
		pass

	def record_epoch_end(self, epoch, val_stat,  test_stat, **kwargs):
		entry = [self[k] for k in self.weights.keys()]
		if 'class_norms' in kwargs:
			class_norm_entry = [v for _, v in kwargs['class_norms'].items()]
		else:
			class_norm_entry = [1.0]*len(self.weights)
		self.class_norm_logs.append(class_norm_entry)
		# Place the statistic to record in the final position
		entry.extend([val_stat, test_stat])
		self.result_logs.append(entry)

	def viz_results(self, save_loc, group_aux=True):
		to_viz_classnorms = np.array(self.class_norm_logs)
		to_viz = np.array(self.result_logs)
		all_keys = list(self.weights.keys())
		prim_idx = all_keys.index(self.prim_key)
		prim_vals = to_viz[:, prim_idx]
		fig, ax = plt.subplots(1, 2, figsize=(16, 8))
		ax[0].plot(range(len(prim_vals)), prim_vals, label='Primary Task Weighting')
		for idx_, key in enumerate(all_keys):
			if idx_ == prim_idx:
				desc = '{} Norm'.format(key) if not group_aux else 'Norm Per-Auxiliary Task'
				ax[1].plot(range(len(prim_vals)), to_viz_classnorms[:, idx_], linestyle='-.', label=desc)
				continue
			desc = '{}'.format(key) if not group_aux else 'Weight Per-Auxiliary Task'
			ax[0].plot(range(len(prim_vals)), to_viz[:, idx_], linestyle='dashed', label=desc)
			desc = '{} Norm'.format(key) if not group_aux else 'Norm Per-Auxiliary Task'
			ax[1].plot(range(len(prim_vals)), to_viz_classnorms[:, idx_], linestyle='-.', label=desc)
			if group_aux:
				break
		for i in range(2):
			ax[i].set_xlabel('Epoch')
			ax[i].set_ylabel('Weighting')
			ax[i].legend(loc='lower left')
		ax2 = ax[0].twinx()
		ax2.plot(range(len(prim_vals)), to_viz[:, -1], color='tab:red', label='Test Accuracy')
		ax2.plot(range(len(prim_vals)), to_viz[:, -2], color='tab:cyan', label='Val Accuracy')
		min_, max_ = np.min(to_viz[:, -2:]) - 0.01, np.max(to_viz[:, -2:]) + 0.01
		ax2.set_ylim(min_, max_)
		ax2.set_ylabel('Test/Val Accuracy', color='tab:red')
		ax2.legend(loc='upper left')
		plt.tight_layout()
		plt.savefig('{}/weighting_vrs_stat.png'.format(save_loc))


class MetaWeighter(Weighter):
	def __init__(self, prim_key, aux_keys, meta_lr_weight=1e-2):
		super(MetaWeighter, self).__init__(prim_key, aux_keys)
		# Finished setting up here
		# perform approriate intialization here
		all_classes = [prim_key, *aux_keys]
		self.weights = self.create_weights(all_classes)
		self.meta_lr_weight = meta_lr_weight
		self.is_meta = True

	def create_weights(self, classes, norm=1.0, requires_grad=True):
		inits = np.array([1.0]* len(classes))
		if norm > 0:
			inits = norm * inits / (sum(inits))
		# Create the weights
		weights = {class_: torch.tensor([inits[id_]]).float().cuda() for id_, class_ in enumerate(classes)}
		if requires_grad:
			for _, v in weights.items():
				v.requires_grad = True
		return weights
	
	def __getitem__(self, key):
		if not hasattr(self, 'sm_weights'):
			self.sm_weights = self.get_softmax()
		return self.sm_weights[key]
	
	# Do softmax on the meta-weights
	def get_softmax(self):
		with torch.no_grad():
			keys = []
			values = []
			for k, v in self.weights.items():
				keys.append(k)
				values.append(v)
			joint_vec = torch.cat(values)
			softmax = F.softmax(joint_vec, dim=-1)
		return {k: v for k, v in zip(keys, softmax)}
	
	# Perform update on meta-related weights.
	def update_meta_weights(self):
		# Todo [ldery] - consider clipping the norm of the gradients if you're not doing cosine
		meta_weights = self.weights
		with torch.no_grad():
			for key in meta_weights.keys():
				if meta_weights[key].grad is None:
					meta_weights[key].grad = torch.zeros_like(meta_weights[key])
				new_val = meta_weights[key] - (self.meta_lr_weight * meta_weights[key].grad)
				meta_weights[key].copy_(new_val)
				meta_weights[key].grad.zero_()

		self.sm_weights = self.get_softmax()

CONFIGS = {
	0: {
		'TAPT-MLM': 1.0/3,
		'DAPT-MLM': 1.0/3,
		'citation_intent': 1.0/3,
		'chemprot': 1.0/3,
		'sciie': 1.0/3,
	},
	1: {
		'TAPT-MLM': 0.333,
		'DAPT-MLM': 0.167,
		'citation_intent': 0.5,
		'chemprot': 0.5,
		'sciie': 0.5,
	},
	2: {
		'TAPT-MLM': 0.5,
		'DAPT-MLM': 0.167,
		'citation_intent': 0.333,
		'chemprot': 0.333,
		'sciie': 0.333,
	},
	3: {
		'TAPT-MLM': 0.4,
		'DAPT-MLM': 0.2,
		'citation_intent': 0.4,
		'chemprot': 0.4,
		'sciie': 0.4,
	}
}

class DefaultWeighter(Weighter):
	def __init__(self, prim_key, aux_keys, init_val=1.0, load_config=None):
		init_val = 1.0 / (1 + len(aux_keys))
		super(DefaultWeighter, self).__init__(prim_key, aux_keys, init_val=init_val)
		self.config = None
		if load_config is not None:
			assert isinstance(load_config, int), 'Load Config must be an integer'
			self.config = CONFIGS[load_config]
		print('this is the loaded config : ', self.config)

	def __getitem__(self, key):
		if self.config is not None:
			return self.config[key]
		else:
			return self.init_val

	def prep_epoch_start(self, epoch, **kwargs):
		pass


class AlternatingWeighter(Weighter):
	def __init__(self, prim_key, aux_keys, init_val=1.0, alt_freq=1, prim_start=0):
		super(AlternatingWeighter, self).__init__(prim_key, aux_keys, init_val=init_val)
		self.alt_freq = alt_freq
		self.prim_start = prim_start

	def prep_epoch_start(self, epoch, **kwargs):
		# restore the initial weighting
		if (epoch < self.prim_start) or (((epoch - self.prim_start) // self.alt_freq) % 2) == 0:
			prim_val = 0.0
			aux_val = self.init_val
		else:
			aux_val = 0.0
			prim_val = self.init_val
		self.weights = {k: aux_val for k in self.aux_keys}
		self.weights[self.prim_key] = prim_val

class PhaseIn(Weighter):
	def __init__(self, prim_key, aux_keys, init_val=1.0, prim_start=10, max_epochs=100):
		super(PhaseIn, self).__init__(prim_key, aux_keys, init_val=init_val)
		self.prim_start = prim_start
		self.delta = init_val / (max_epochs - prim_start)

	def prep_epoch_start(self, epoch, **kwargs):
		if (epoch < self.prim_start):
			prim_val = 0.0
			aux_val = self.init_val
		else:
			prim_val = (epoch - self.prim_start) * self.delta
			aux_val = self.init_val - prim_val
		self.weights = {k: aux_val for k in self.aux_keys}
		self.weights[self.prim_key] = prim_val


class WarmUpAndDown(Weighter):
	def __init__(self, prim_key, aux_keys, init_val=0.0, prim_first=True, end_val=1.0, reset_every=5):
		super(WarmUpAndDown, self).__init__(prim_key, aux_keys, init_val=init_val)
		self.end_val = end_val
		self.reset_every = reset_every
		self.prim_first = prim_first
		self.delta = (end_val - init_val) / self.reset_every
		assert self.delta > 0, 'End - {} is less than start - {}'.format(end_val, init_val)

	def prep_epoch_start(self, epoch, **kwargs):
		# We are starting the epoch.
		eff_epoch = epoch // self.reset_every
		aux_delt, prim_delt = self.delta, self.delta
		if self.prim_first:
			if eff_epoch % 2 == 0:
				# We are in a phase where primary weight is dropping
				prim_delt = -self.delta
			else:
				aux_delt = -self.delta
		else:
			# We are in a phase where auxiliary weight is dropping
			if eff_epoch % 2 == 0:
				aux_delt = -self.delta
			else:
				prim_delt = -self.delta
		assert prim_delt * aux_delt < 0, 'prim_delt = {} and aux_delt = {} are same sign'.format(prim_delt, aux_delt)
		if epoch % self.reset_every == 0:
			# We are resetting
			self.weights[self.prim_key] = self.end_val if prim_delt < 0 else self.init_val
			for k, v in self.weights.items():
				if k == self.prim_key:
					continue
				self.weights[k] = self.end_val if aux_delt < 0 else self.init_val
		else:
			# We are doing a normal update
			old_val = self.weights[self.prim_key]
			self.weights = {k: self.weights[k] + aux_delt for k in self.aux_keys}
			self.weights[self.prim_key] = old_val + prim_delt

