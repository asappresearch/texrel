class Stats(object):
    def __init__(self, keys):
        self._keys = keys
        self.reset()

    def __repr__(self):
        res = ''
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            res += '  ' + str(k) + ':' + str(v) + '\n'
        return res

    def reset(self):
        for k in self._keys:
            self.__dict__[k] = 0.0

    def __iadd__(self, second):
        assert isinstance(second, Stats)
        for k, v in second.__dict__.items():
            if k.startswith('_'):
                continue
            self.__dict__[k] += second.__dict__[k]
        return self
