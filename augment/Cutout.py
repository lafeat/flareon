import numpy as np
import torch

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, ten):
        """
        Args:
            img (Tensor): Tensor image of size (N, C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        n = ten.size(0)
        h = ten.size(2)
        w = ten.size(3)

        for i in range(n):
            mask = np.ones((h, w), np.float32)
            for n in range(self.n_holes):
                # (x,y)表示方形补丁的中心位置
                y = np.random.randint(h)
                x = np.random.randint(w)
                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)
                mask[y1: y2, x1: x2] = 0.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(ten[i]).cuda()
            ten[i] = ten[i] * mask

        return ten