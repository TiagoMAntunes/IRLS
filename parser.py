import numpy as np

class Parser:
    def __init__(self, target):
        with open(target) as f:
            self._lines = f.readlines()
    
    def parse(self):
        if not hasattr(self, '_data'):
            f = lambda a: (int(a[0]), list(map(lambda x: (int(x[0]),int(x[1])), map(lambda x: x.split(':'), a[1:]))))
            self._data = [f(x.split()) for x in self._lines]
        return self._data

    def transform_x(self):
        self.parse()
        contents = list(map(lambda x: x[1], self._data))
        size = 124
        def transform(l):
            line = [0 for i in range(size)]
            line[0] = 1
            for i in l:
                line[i[0]] = i[1]
            return line
        p = list(map(transform, contents))
        return np.array(p)

    def transform_y(self):
        self.parse()
        contents = list(map(lambda x: x[0], self._data))
        return (np.array(contents) + 1) // 2 # transform into {0, 1}