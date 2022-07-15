from torchvision.datasets import VisionDataset
from scipy.stats import beta, uniform

import distortions


class AugmentedDataset(VisionDataset):
    """
    Distortions take as argument two images and an additional parameter.
    The parameter is patch_size in the case of patch-shuffling and the
    occlusion proportion for occluding methods.
    To sample the occlusion proportion from the beta distribution, set
    ```sample``` to True. The ```parameter``` field will be used for the
    Beta distribution.

    Args:
        dataset: data set to modify
        distortion: distortion to apply to data set
        parameter: parameter for the distortion method
        imbalanced: sample from an imbalanced beta distribution
        sample: use the distribution parameter to sample the beta distribution
    """
    def __init__(self, dataset, distortion, parameter, imbalanced=False, sample=False):
        self.parameter = parameter
        self.dataset = dataset
        self.distortion = distortion
        self.imbalanced = int(imbalanced)
        self.sample = sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img1, target = self.dataset[index][0], self.dataset[index][1]
        if self.sample:
            lam = beta.rvs(self.parameter + self.imbalanced, self.parameter)
            img = getattr(distortions, self.distortion)(img1, None, lam)
        else:
            img = getattr(distortions, self.distortion)(img1, None, self.parameter)
        return img, target
