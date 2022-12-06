import numpy as np
import torch

import os
import sys
print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


TIME_UNITS = [
    's',  # Seconds
    'ms', # Mili-seconds
    'us', # Micro-seconds
    'ns'  # Nano-seconds
]

class Time:
    """This class has [n]-size time vector with specific unit-base.\n
        Available `unit` arguments are
        - `s` : Seconds
        - `ms`: Mili-seconds
        - `us`: Micro-seconds
        - `ns`: Nano-seconds
        """
    def __init__(self, data:torch.Tensor, unit:str) -> None:
        assert (data.ndim == 1), 'Time stamp must be 1-D vector'
        assert (unit in TIME_UNITS), 'Unvalid time unit'

        self.single = True if data.size(0) == 1 else False
        self.unit = unit

        if self.single:
            # self.data = data.cpu()
            raise AssertionError('Not supported yet!')
        else:
            self.cudable = torch.cuda.is_available() # 이 브랜치를 타는 경우에만 Member variable 생성
            self.data = data.cuda() if self.cudable else data.cpu()

    def __str__(self) -> str:
        return str(self.data) + ', [%s]'%self.unit

    def to(self, unit:str):
        assert (unit in TIME_UNITS), 'Unvalid time unit'

        def idx(_unit):
            if _unit == 'ns':
                return 0
            elif _unit == 'us':
                return 1
            elif _unit == 'ms':
                return 2
            elif _unit == 's':
                return 3

        from_ = idx(self.unit)
        to_ = idx(unit)
        offset = (1e3)**(to_-from_)
        # print('%s to %s: offset: %e' % (self.unit, unit, offset))

        self.data *= offset
        self.unit = unit
        return self

    @property
    def length(self):
        return self.data.size(0)

if __name__ == '__main__':

    t = torch.arange(0, 1, 0.1).cuda()
    print(t)

    t = Time(t, 'ms')
    print(t)

    t = t.to('s')
    print(t)

    t = t.to('ns')
    print(t)


