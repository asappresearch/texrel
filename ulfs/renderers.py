import torch
import matplotlib.pyplot as plt


def render_q(world, q_learner):
    qvalues_list = []
    for h in range(world.height):
        row = []
        qvalues_row = []
        for w in range(world.width):
            loc = torch.LongTensor([h, w])
            c = world.get_at(loc)
            # print('c', c)
            if c == '' or c == 'A':
                world.set_agent_loc(loc)
                s = world.get_state()
                s = s.view(1, *s.size())
                qvalues = q_learner(s)
                # print('qvalues', qvalues)
                qvalue, a = qvalues.data.max(1)
                qvalue = qvalue[0]
                qvalues_row.append(qvalue)
                a = a[0]
                # print('a', a)
                act, direction = world.render_action(a)
                # print(act, dir)
                if act == 'M':
                    row.append(direction)
                elif act == 'C':
                    row.append('c')
                elif act == 'E':
                    row.append('e')
            else:
                row.append(c)
                qvalues_row.append(0)
        print(''.join(row))
        qvalues_list.append(qvalues_row)
    print('')
    for row in qvalues_list:
        print(' '.join(['%.2f' % q for q in row]))


def render_policy(world, state_features, policy):
    for h in range(world.height):
        row = []
        for w in range(world.width):
            loc = torch.LongTensor([h, w])
            c = world.get_at(loc)
            if c == '' or c == 'A':
                world.set_agent_loc(loc)
                s = world.get_state()
                s = s.view(1, *s.size())
                global_idxes = torch.LongTensor([[0]])
                phi = state_features(s)
                action_probs_l, elig_idxes_pairs, actions, entropy, (
                    matches_argmax_count, stochastic_draws_count) = policy(
                        phi, global_idxes=global_idxes)

                # print('actions[0]', actions[0])
                # asdf
                a = actions[0][0]
                act, direction = world.render_action(a)
                if act == 'M':
                    row.append(direction)
                elif act == 'C':
                    row.append('c')
                elif act == 'E':
                    row.append('e')
            else:
                row.append(c)
        print(''.join(row))


def render_v(value_png, world, state_features, value_function):
    """
    assumptions on world. has following properties:
    - height
    - width
    - locs (location of various objects, like food)
    - reps
    and methods:
    - get_at(loc)
    - set_agent_loc(loc)
    - get_state()
    - render()

    state_features is a net, that takes a world state, and generates features
    value_function takes state_features, and outputs a value
    value_png is path to generated png file
    """
    v_matrix = torch.FloatTensor(world.height, world.width).fill_(0)
    for h in range(world.height):
        row = []
        for w in range(world.width):
            loc = torch.LongTensor([h, w])
            c = world.get_at(loc)
            if c == '' or c == 'A':
                world.set_agent_loc(loc)
                s = world.get_state()
                s = s.view(1, *s.size())

                phi = state_features(s)
                v = value_function(phi).data[0][0]
                v_matrix[h, w] = v
                row.append('%.3f' % v)
            else:
                row.append(c)
        print(' '.join(row))
        # values_list.append(values_row)
    print('')

    world.render()
    print('v_matrix', v_matrix)
    plt.clf()
    plt.imshow(v_matrix.numpy(), interpolation='nearest')
    for loc in world.locs:
        _loc = loc['loc']
        p = loc['p']
        plt.text(_loc[1], _loc[0], '[%s:%s]' % (p, world.reps[p]))
    plt.colorbar()
    plt.savefig(value_png)

    # for row in values_list:
    #     print(' '.join(['%.2f' % q for q in row]))
