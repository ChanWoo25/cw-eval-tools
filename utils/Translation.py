import numpy as np
import torch
from typing import Tuple


import os
import sys
print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class Translation:
    """Translation tensor class."""

    def __init__(self, data:torch.Tensor) -> None:
        self.single:bool
        self.data:torch.Tensor = data
        self.cudable:bool = torch.cuda.is_available()

        assert (type(data) is torch.Tensor), ('data type must be torch.Tensor')

        if len(self.data.shape) == 2 and self.data.shape[-1] == 3:
            self.single = False
        elif len(self.data.shape) == 1 and self.data.shape[-1] == 3:
            self.single = True
            print('Single translation is not yet supported!')
        else:
            raise AssertionError('translation vector size must be [n, 3] or [3]')

        if self.cudable:
            self.data = self.data.cuda()

    def length(self):
        if self.single:
            return 1
        else:
            return self.data.shape[0]

    def shape(self):
        return self.data.shape

    def numpy(self) -> np.ndarray:
        return self.data.cpu().numpy()

if __name__ == '__main__':

    t1 = Translation(torch.randn(2, 3))
    t2 = Translation(torch.randn(2, 3))
    print(t1.data)
    print(t2.data)


