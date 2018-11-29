import numpy as np
import h5features
import random


if __name__=='__main__':
  
    path = 'random.h5f'

    features = []
    items = ['file1', 'file2', 'file3', 'file4', 'file5']

    times = [] #[np.arange(features[0].shape[0], dtype=np.float32) * 0.01 + 0.0025]
    for item in items:
        features.append(np.random.rand(random.randint(50, 150), 40))
        times.append(np.arange(features[-1].shape[0], dtype=np.float32) * 0.01 + 0.0025)

    h5features.write(path, '/features/', items, times, features)
