from torch.utils.data import Dataset
import h5py
import torch

from einops import rearrange
import numpy as np

class Shapes3D():
    def __init__(self, root, phase):
        assert phase in ['train', 'val', 'test']
        with h5py.File(root, 'r') as f:
            if phase == 'train':
                self.imgs = f['images'][:400000]
            elif phase == 'val':
                self.imgs = f['images'][400001:430000]
            elif phase == 'test':
                self.imgs = f['images'][430001:460000]
            else:
                raise NotImplementedError

    def __getitem__(self, index):
        img = self.imgs[index]
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img.float() / 255.

        return img

    def __len__(self):
        return len(self.imgs)

class DebugShapes3D():
    def __init__(self, root, phase):
        assert phase in ['train', 'val', 'test']
        with h5py.File(root, 'r') as f:
            if phase == 'train':
                self.imgs = f['images'][:400]
            elif phase == 'val':
                self.imgs = f['images'][401:430]
            elif phase == 'test':
                self.imgs = f['images'][431:460]
            else:
                raise NotImplementedError

    def __getitem__(self, index):
        img = self.imgs[index]
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img.float() / 255.

        return img

    def __len__(self):
        return len(self.imgs)


class DataLoader():
    def __init__(self, dataset, batch_size, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = len(self.dataset) // self.batch_size
        self.counter = 0

    def get_batch(self):
        indices = np.random.choice(len(self.dataset), 
            size=self.batch_size, replace=False)
        batch = self.dataset.imgs[indices]
        batch = rearrange((batch.astype(np.float32) / 255.0), 'b h w c -> b c h w')
        return batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < len(self):
            self.counter += 1
            return self.get_batch()
        raise StopIteration

    def __len__(self):
        return self.num_batches





# class WhiteBallDataLoader():
#   def __init__(self, h5):
#     self.h5 = h5
#     assert 'observations' in self.h5.keys() and 'actions' in self.h5.keys()

#   def normalize_actions(self, act_batch):
#     # normalize actions from [0, 5] to [-1, 1]
#     act_batch = (act_batch * 2./5) - 1
#     return act_batch

#   def get_batch(self, batch_size, num_frames):
#     batch_indices = np.random.choice(self.h5['observations'].shape[0], size=batch_size, replace=False)
#     obs_batch = self.h5['observations'][sorted(batch_indices), :num_frames]
#     obs_batch = utils.normalize(obs_batch)
#     obs_batch = einops.rearrange(obs_batch, '... c h w -> ... h w c')
#     obs_batch = tf.convert_to_tensor(obs_batch)
#     if num_frames > 1:
#       act_batch = self.h5['actions'][sorted(batch_indices), :num_frames]
#       act_batch = self.normalize_actions(act_batch)
#       act_batch = tf.convert_to_tensor(act_batch)

#       is_first_batch = np.zeros(act_batch.shape[:-1], np.bool)
#       is_first_batch[:, 0] = True

#       return {'image': obs_batch, 'action': act_batch, 'is_first': is_first_batch}
#     else:
#       obs_batch = einops.rearrange(obs_batch, 'b t ... -> (b t) ...')
#       return {'image': obs_batch}