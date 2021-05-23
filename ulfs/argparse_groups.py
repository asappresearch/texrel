import argparse
from collections import defaultdict


class Group(object):
    def __init__(self, name, parent, parser):
        self.name = name
        self.parent = parent
        self.parser = parser
        self.keys = set()

    def add_argument(self, typed_key, *args, **kwargs):
        prog_key = typed_key.replace('--', '').replace('-', '_')
        assert prog_key not in self.keys
        self.parser.add_argument(typed_key, *args, **kwargs)
        self.keys.add(prog_key)
        self.parent.register_arg(prog_key, self)

    def __str__(self):
        res = 'Group(name=' + self.name + ', keys=' + str(list(self.keys)) + ')'
        return res


class GroupedArgumentParser(object):
    def __init__(self):
        self.groups = {}
        self.parser = argparse.ArgumentParser()
        self.group_by_arg = {}

    def add_group(self, name):
        assert name not in self.groups
        group = Group(name=name, parent=self, parser=self.parser)
        self.groups[name] = group
        return group

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self, args=None):
        # for name, group in self.groups.items():
        #     print(name, group)
        args = self.parser.parse_args(args)
        # root_args = {}
        groups = defaultdict(dict)
        other_args = {}
        for k, v in args.__dict__.items():
            # print('k', k, 'v', v)
            if k in self.group_by_arg:
                group_parser = self.group_by_arg[k]
                # print('group_parser', group_parser)
                group_name = group_parser.name
                groups[group_name][k] = v
            else:
                other_args[k] = v
        if len(groups) == 0:
            return other_args
        # if len(groups) > 0 and len(other_args) == 0:
        #     return groups
        # if len(groups) == 0 and len(other_args) > 0:
        #     return other_args
        groups['root'] = other_args
        return groups
        # return other_args, groups

    def register_arg(self, arg, group):
        assert arg not in self.group_by_arg
        # print('register_arg', arg, group)
        self.group_by_arg[arg] = group
