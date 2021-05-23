class Params(object):
    def __init__(self, dict={}):
        for k, v in dict.items():
            self.__dict__[k] = v

    def __repr__(self):
        res = ''
        for k, v in sorted(self.__dict__.items()):
            res += '  ' + str(k) + ':' + str(v) + '\n'
        return res

    def extract_from_args(self, args, params_keys):
        for k in params_keys:
            if k in args.__dict__:
                self.__dict__[k] = args.__dict__[k]
                del args.__dict__[k]
        return self

    def __contains__(self, key):
        return key in self.__dict__
