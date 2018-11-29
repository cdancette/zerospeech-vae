import h5features
import h5py
import random
import numpy as np
import torch
import torch.utils.data 

class FeaturesDataset(torch.utils.data.Dataset):
	def __init__(self, features):
		super(FeaturesDataset, self).__init__()
		self.features = features #h5py.File(h5features)['features']
		#reader = .reader.Read(h5features_path)
		#data = reader.read()

	def __len__(self):
		#return len(self.features['features'])
		return len(self.features)

	def __getitem__(self, idx):
		return torch.FloatTensor(self.features[idx])

def get_train_test_datasets(h5feature_path, train_ratio):
	reader = h5features.reader.Reader(h5feature_path)
	data = reader.read()
	dict_features = data.dict_features()
	files = list(data.dict_features())
	random.shuffle(files)
	train_idx = int(len(files) * train_ratio)

	train_dataset = np.vstack(dict_features[f] for f in files[:train_idx])
	test_dataset = np.vstack(dict_features[f] for f in files[train_idx:])
	return FeaturesDataset(train_dataset), FeaturesDataset(test_dataset)


def get_dataset(h5feature_path):
	reader = h5features.reader.Reader(h5feature_path)
	data = reader.read()
	dict_features = data.dict_features()

	dataset = np.vstack(dict_features[f] for f in dict_features)
	return FeaturesDataset(dataset)
