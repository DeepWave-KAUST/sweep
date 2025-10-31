import jax
import jax.numpy as jnp
from .utils import split_model, split_by_lr

class DataLoader:

    def __init__(self, obs, sources, receivers, batch_size=8, steps_per_epoch=1, mask=None, models=None, wavelets=None, key=None, model_split=False):

        self.obs = obs
        self.sources = sources
        self.receivers = receivers
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.mask = mask
        self.models = models
        self.key = key
        self.model_split = model_split
        self.wavelets = wavelets

        self.check()
        self.init()

    def check(self):
        if self.model_split and (self.models is None or not isinstance(self.models, (list, tuple))):
            raise ValueError("When model_split is True, models must be provided as a list or tuple.")

    def init(self,):
        """Split the model parameters into different groups based on the geometry of the data."""
        if self.model_split:
            self.msplit, self.left, self.right, self.sources, self.receivers = split_model(self.models, self.sources, self.receivers, one_side_expand=50)
            self.shape = self.msplit[0].shape[-2:]
        else:
            self.left, self.right = None, None
            self.shape = None
    @property
    def device_count(self):
        return jax.device_count()
    
    @property
    def multi_gpu(self):
        return self.device_count > 1

    @property
    def has_wavelets(self):
        return self.wavelets is not None
    
    @property
    def shotwise_wavelets(self):
        if self.has_wavelets:
            return self.wavelets.shape[0] == self.obs.shape[0]
        else:
            return False

    def __len__(self):
        return self.obs.shape[0]
    
    def __iter__(self):
        return self
    
    def __next__(self):
        raise NotImplementedError("Use the 'next' method to get batches from the DataLoader.")
    
    @property
    def use_all_shots(self):
        """Check if all shots are used in the DataLoader."""
        if isinstance(self.batch_size, int):
            return self.batch_size >= len(self)
        elif isinstance(self.batch_size, (list, tuple)):
            return max(self.batch_size) >= len(self)
    
    def get_batchsize(self, freq_idx):
        """Get the batch size based on the frequency index"""
        batchsize_this_epoch = 0
        if isinstance(self.batch_size, int):
            batchsize_this_epoch = self.batch_size                
        elif isinstance(self.batch_size, (list, tuple)):
            batchsize_this_epoch = self.batch_size[freq_idx]
        assert batchsize_this_epoch > 0, "Batch size must be greater than 0."

        return batchsize_this_epoch
    
    def sample_shots_randomly(self, batch_size=None):
        """Sample shots randomly from the dataset.

        Args:
            batch_size (int, optional): The number of shots to sample. If None, use the default batch size.
        """

        batch_size = self.batch_size if batch_size is None else batch_size

        self.key, subkey = jax.random.split(self.key)

        # Randomly sample shots from range (0, nshots)
        rand_shots = jax.random.permutation(subkey, len(self))[:batch_size] if not self.use_all_shots else jnp.arange(len(self))
        # All the random shots will be distributed across devices,
        # Each device will run only one shot at a time.
        self.steps_per_epoch = batch_size//self.device_count if self.multi_gpu else self.steps_per_epoch
        self.rand_shots = rand_shots.reshape(self.steps_per_epoch, self.device_count, -1)
        self.shots_per_gpu = self.rand_shots.shape[-1]

    def next(self, step):

        obs_batch = self.obs[self.rand_shots[step]]
        sources_batch = self.sources[self.rand_shots[step]]
        receivers_batch = self.receivers[self.rand_shots[step]]
        if self.mask is not None:
            mask_batch = self.mask[self.rand_shots[step]]
        else:
            mask_batch = None

        if self.model_split:
            rs = self.rand_shots[step].flatten()
            models_batch = [jnp.stack([m[:, l:r] for l, r in zip(self.left[rs], self.right[rs])]).reshape(self.device_count, -1, *self.shape) for m in self.models]
        else:
            models_batch = [jnp.tile(m.reshape(1, 1, *m.shape), (self.device_count, self.shots_per_gpu, 1, 1)) for m in self.models]

        if self.shotwise_wavelets:
            wavelets_batch = self.wavelets[self.rand_shots[step]]
        else:
            wavelets_batch = self.wavelets.repeat(self.shots_per_gpu, axis=0)[None, ...].repeat(self.device_count, axis=0)

        return obs_batch, sources_batch, receivers_batch, mask_batch, models_batch, wavelets_batch