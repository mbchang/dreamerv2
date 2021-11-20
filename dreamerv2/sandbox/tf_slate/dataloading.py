from torch.utils.data import Dataset
import h5py
import torch

import einops
import numpy as np
import pathlib
import tensorflow as tf

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

    # [0, 1] --> [0, 1]
    def normalize_obs(self, x):
        return x

    # [0, 1] --> [0, 1]
    def unnormalize_obs(self, x):
        return tf.clip_by_value(x, 0., 1.)

    def get_batch(self):
        indices = np.random.choice(len(self.dataset), 
            size=self.batch_size, replace=False)
        batch = self.dataset.imgs[indices]
        batch - self.normalize_obs(batch)
        batch = einops.rearrange((batch.astype(np.float32) / 255.0), 'b h w c -> b c h w')
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


class WhiteBallDataLoader():
    def __init__(self, h5, batch_size):
        self.h5 = h5
        assert 'observations' in self.h5.keys() and 'actions' in self.h5.keys()

        self.batch_size = batch_size

    def normalize_obs(self, x):
        return x - 0.5 # Rescale to [-0.5, 0.5]

    def unnormalize_obs(self, x):
        """Renormalize from [-0.5, 0.5] to [0, 1]."""
        return tf.clip_by_value(x + 0.5, 0., 1.)

    # # [0, 1] --> [0, 1]
    # def normalize_obs(self, x):
    #     return x

    # # [0, 1] --> [0, 1]
    # def unnormalize_obs(self, x):
    #     return tf.clip_by_value(x, 0., 1.)

    def normalize_actions(self, act_batch):
        # normalize actions from [0, 5] to [-1, 1]
        act_batch = (act_batch * 2./5) - 1
        return act_batch

    def get_batch(self, num_frames=1):
        batch_indices = np.random.choice(self.h5['observations'].shape[0], size=self.batch_size, replace=False)
        obs_batch = self.h5['observations'][sorted(batch_indices), :num_frames]
        obs_batch = self.normalize_obs(obs_batch)
        # obs_batch = einops.rearrange(obs_batch, '... c h w -> ... h w c')
        # obs_batch = tf.convert_to_tensor(obs_batch)
        if num_frames > 1:
            act_batch = self.h5['actions'][sorted(batch_indices), :num_frames]
            act_batch = self.normalize_actions(act_batch)
            # act_batch = tf.convert_to_tensor(act_batch)

            is_first_batch = np.zeros(act_batch.shape[:-1], np.bool)
            is_first_batch[:, 0] = True

            return {'image': obs_batch, 'action': act_batch, 'is_first': is_first_batch}
        else:
            obs_batch = einops.rearrange(obs_batch, 'b t ... -> (b t) ...')
            return {'image': obs_batch}


class DMCDatLoader():
    def __init__(self, dataroot, batch_size):
        self.episodes = DMCDatLoader.load_episodes(pathlib.Path(dataroot) / 'train_episodes')
        self.batch_size = batch_size

    def normalize_obs(self, x):
        return x - 0.5 # Rescale to [-0.5, 0.5]

    def unnormalize_obs(self, x):
        """Renormalize from [-0.5, 0.5] to [0, 1]."""
        # return x  + 0.5
        return tf.clip_by_value(x + 0.5, 0., 1.)

    # taken from dreamer/replay
    @staticmethod
    def load_episodes(directory, capacity=None, minlen=1):
        # The returned directory from filenames to episodes is guaranteed to be in
        # temporally sorted order.
        filenames = sorted(directory.glob('*.npz'))
        if capacity:
            num_steps = 0
            num_episodes = 0
            for filename in reversed(filenames):
                length = int(str(filename).split('-')[-1][:-4])
                num_steps += length
                num_episodes += 1
                if num_steps >= capacity:
                    break
            filenames = filenames[-num_episodes:]
        episodes = {}
        for filename in filenames:
            try:
                with filename.open('rb') as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f'Could not load episode {str(filename)}: {e}')
                continue
            episodes[str(filename)] = episode
        return episodes

    def get_batch(self, num_frames=1):
        obs_batch = []
        act_batch = []

        episodes = list(self.episodes.values())
        for i in range(self.batch_size):
            # sample random episode
            episode = episodes[np.random.randint(len(episodes))]
            total_length = len(episode['action'])
            assert total_length > num_frames

            # sample random chunk
            start = np.random.randint(total_length-num_frames+1)
            end = start + num_frames

            # append
            obs_batch.append(episode['image'][start:end])
            act_batch.append(episode['action'][start:end])  # drop first action

        # stack
        obs_batch = np.stack(obs_batch)
        act_batch = np.stack(act_batch)

        # reshape
        obs_batch = einops.rearrange(obs_batch, '... h w c -> ... c h w')

        # normalize
        obs_batch = self.normalize_obs(obs_batch.astype(np.float32) / 255.0)

        if num_frames == 1:
            batch =  {
            'image': np.squeeze(obs_batch, 1), 
            'action': np.squeeze(act_batch, 1)}
        else:
            batch =  {'image': obs_batch, 'action': act_batch}

        return batch

