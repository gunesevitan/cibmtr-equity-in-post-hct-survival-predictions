import torch
from torch.utils.data import Dataset


class TabularInMemoryDataset(Dataset):

    def __init__(self, features, targets=None):

        self.features = features
        self.targets = targets

    def __len__(self):

        """
        Get the length the dataset

        Returns
        -------
        length: int
            Length of the dataset
        """

        return len(self.features)

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx: int
            Index of the sample (0 <= idx < length of the dataset)

        Returns
        -------
        features: torch.Tensor of shape (n_features)
            Features tensor

        targets: torch.Tensor of shape (n_targets)
            Targets tensor
        """

        features = self.features[idx]
        features = torch.as_tensor(features, dtype=torch.float)

        if self.targets is not None:

            targets = self.targets[idx]
            targets = torch.as_tensor(targets, dtype=torch.float)

            return features, targets

        else:

            return features
