"""
given an encoded relation, and an example of this relation (or not), can we classify
it correctly?

means, no need for any nlp bits, or REINFORCE.

let's us test various ways of using the encoded relation to classify the image

a few ways we can try:
- concatenate the encoding with that of the image
- distance of encoding from image encoding (eg "dot product is positive or negative?" perhaps?)
- attention over output feature planes
- attention over filter planes at each layer
- mapping from encoded relation to the filter weights of each layer :P
  (this means we wont backprop so much onto the weights, as through the weights)
"""
import time
import json

import torch
from torch import nn, optim

from ulfs.stats import Stats
from ulfs.runner_base_v1 import RunnerBase
from ulfs import utils

from ref_task import params_groups
from ref_task.models import receiver_model as receiver_model_lib
from ref_task import dataset_family_lib


class Runner(RunnerBase):
    def __init__(self):
        super().__init__(
            save_as_statedict_keys=['model', 'opt'],
            step_key='episode'
        )

    def setup(self, p):
        params_groups.process_args(args=p)

        print('p', self.params)

        print('')
        print('ref', p.ref)
        print('')

        self.dataset = dataset_family_lib.build_family(**utils.filter_dict_by_prefix(p.__dict__, prefix='ds_'))
        self.ds_meta = self.dataset.meta
        print('ds_meta', json.dumps(self.ds_meta.__dict__, indent=2))

        self.vocab_size = self.ds_meta.vocab_size
        self.grid_planes = self.ds_meta.grid_planes

        p.grid_size = self.ds_meta.grid_size
        p.max_colors = self.ds_meta.num_colors
        p.max_shapes = self.ds_meta.num_shapes
        self.in_seq_len = self.ds_meta.utt_len
        print('in seq len', self.in_seq_len)

        print('params', self.params)
        print('vocab_size', self.vocab_size)

        self.model = receiver_model_lib.build_receiver_model(
            params=p,
            ds_meta=self.ds_meta,
            vocab_size=self.ds_meta.vocab_size,
            utt_len=self.ds_meta.utt_len
        )

        if p.enable_cuda:
            self.model = self.model.cuda()

        print('receiver model', self.model)

        self.opt = optim.Adam(lr=0.001, params=self.model.parameters())
        self.crit = nn.CrossEntropyLoss(reduction='none')
        self.stats = Stats([
            'episodes_count',
            'loss_sum',
            'acc_sum'
        ])

    def step(self, p):
        episode = self.episode
        render = self.should_render()

        def forward(split_name: str, render=False):
            batch_size = p.batch_size
            b = self.dataset.sample(batch_size=batch_size, split_name=split_name, no_sender=True)
            receiver_examples_t, labels_t, hypotheses_t = map(b.__getitem__, [
                'inner_test_examples_t', 'inner_test_labels_t', 'hypotheses_t'
            ])

            M, N = hypotheses_t.size()
            batch_size = N
            if p.enable_cuda:
                labels_t = labels_t.cuda()
                receiver_examples_t = receiver_examples_t.cuda()
                hypotheses_t = hypotheses_t.cuda()
            labels_t = labels_t.long()

            M, N = hypotheses_t.size()
            relations_onehot_t = torch.zeros(M, N, self.vocab_size, device=hypotheses_t.device)
            relations_onehot_t.scatter_(-1, hypotheses_t.view(M, N, 1), 1.0)
            out = self.model(relations_onehot_t, receiver_examples_t)
            _, pred = out.max(-1)
            correct = pred.view(-1) == labels_t.view(-1)
            acc = correct.long().sum().item() / labels_t.numel()
            crit_loss = self.crit(out.view(-1, out.size(-1)), labels_t.view(-1))

            loss = crit_loss.mean()

            return labels_t, receiver_examples_t, pred, correct, acc, loss

        self.model.train()
        labels_t, receiver_examples_t, pred, correct, acc, loss = forward(
            split_name='train')

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.stats.loss_sum += loss.item()
        self.stats.acc_sum += acc
        self.stats.episodes_count += 1

        terminate_reason = ''

        if p.max_steps is not None and p.max_steps > 0 and episode >= p.max_steps:
            print(f'reached max steps {p.max_steps} => terminating')
            self.finish = True
            terminate_reason = 'max_steps'

        if p.max_mins is not None and time.time() - self.start_time >= p.max_mins * 60:
            print('reached terminate time => terminating')
            self.finish = True
            terminate_reason = 'timeout'

        if render or self.finish:
            stats = self.stats
            log_dict = {
                'loss': stats.loss_sum / stats.episodes_count,
                'acc': stats.acc_sum / stats.episodes_count,
                'eps': stats.episodes_count / (time.time() - self.last_print)
            }

            format_str = (
                'e={episode} '
                'eps={eps:.1f} '
                'loss={loss:.3f} '
                'acc={acc:.3f} '
            )
            for split_name in ['val_same', 'val_new', 'test_same', 'test_new']:
                eval_acc_l = []
                for it in range(1):
                    self.model.eval()
                    labels_t, receiver_examples_t, pred, correct, eval_acc, eval_loss = forward(
                        split_name=split_name, render=(it == 0))
                    eval_acc_l.append(eval_acc)
                log_dict[f'{split_name}_acc'] = sum(eval_acc_l) / len(eval_acc_l)
                format_str += f'{split_name}_acc={{{split_name}_acc:.3f}}'
            self.print_and_log(
                log_dict,
                format_str
            )
            stats.reset()

        if self.finish:
            print('log_dict', log_dict)
            self.res = dict(log_dict)
            self.res.update({
                'batch': episode,
                'elapsed': (time.time() - self.start_time),
                'terminate_reason': terminate_reason
            })
            print('self.res', self.res)


if __name__ == '__main__':
    runner = Runner()

    runner.add_param('--crit-loss-rewards', action='store_true')
    runner.add_param('--max-steps', type=int, default=0)
    runner.add_param('--max-mins', type=float, help='finish running if reach this elapsed time')

    params_groups.add_ds_args(runner)
    params_groups.add_common_args(runner)
    params_groups.add_receiver_args(runner)
    params_groups.add_conv_args(runner)

    runner.parse_args()
    runner.setup_base()
    runner.run_base()
