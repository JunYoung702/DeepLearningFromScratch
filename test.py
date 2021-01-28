import numpy as np
class Test:
    def __init__(self):
        self.x = np.arange(2)

    def func(self):
        return np.sum(self.x)

    def add(self):
        addfunc(self.x, self.func)



def addfunc(x, f):
    x[0] += 1
    x[1] += 1
    print(f())