from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from termcolor import cprint


class GR1DexDatasetImage(BaseImageDataset):
    def __init__(
        self,
        dataset_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name=None,
        use_img=True,
        use_depth=False,
    ):
        super().__init__()
        cprint(f"Loading GR1DexDataset from {dataset_path} horizon:{horizon}", "green")
        self.task_name = task_name
        self.use_img = use_img
        self.use_depth = use_depth

        buffer_keys = [
            "state",
            "action",
        ]

        buffer_keys.append("img")
        buffer_keys.append("wrist_img")

        self.replay_buffer = ReplayBuffer.copy_from_path(dataset_path, keys=buffer_keys)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed
        )
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        cprint(f"Loading GR1DexDataset done", "green")

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        print("==== get_normalizer")
        data = {"action": self.replay_buffer["action"]}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        normalizer["image"] = SingleFieldLinearNormalizer.create_identity()
        normalizer["wrist_image"] = SingleFieldLinearNormalizer.create_identity()

        normalizer["agent_pos"] = SingleFieldLinearNormalizer.create_identity()

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"][:,].astype(np.float32)

        image = sample["img"][:,].astype(np.float32)
        wrist_image = sample["wrist_img"][:,].astype(np.float32)

        data = {
            "obs": {
                "agent_pos": agent_pos,
            },
            "action": sample["action"].astype(np.float32),
        }

        # (16, 512, 512, 3) to (16, 3, 512, 512)
        image = np.transpose(image, (0, 3, 1, 2))
        wrist_image = np.transpose(wrist_image, (0, 3, 1, 2))
        data["obs"]["image"] = image
        data["obs"]["wrist_image"] = wrist_image

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        to_torch_function = lambda x: (
            torch.from_numpy(x) if x.__class__.__name__ == "ndarray" else x
        )
        torch_data = dict_apply(data, to_torch_function)
        return torch_data
