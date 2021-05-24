from ulfs import argparse_groups


def test_argparse_groups():
    grouped_argparse = argparse_groups.GroupedArgumentParser()

    problem = grouped_argparse.add_group('problem')
    problem.add_argument('--num-distractors', type=int, default=0)
    problem.add_argument('--num-input-examples', type=int, default=5)

    receiver = grouped_argparse.add_group('receiver')
    receiver.add_argument('--num-output-fcs', type=int, default=1)

    args = {}
    groups = grouped_argparse.parse_args(args)

    # print('groups.keys()', groups.keys())
    print('groups[problem]', groups['problem'])
    print('groups[receiver]', groups['receiver'])


def test_argparse_groups_nogroups():
    parser = argparse_groups.GroupedArgumentParser()
    parser.add_argument('--num-distractors', type=int, default=0)
    parser.add_argument('--num-input-examples', type=int, default=5)
    parser.add_argument('--num-output-fcs', type=int, default=1)

    args = {}
    args = parser.parse_args(args)
    print('args', args)
