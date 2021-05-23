"""
This task is for training only the sender, no comms, in a supervised manner

for debug running, eg:

python ref_task/run_sender.py --disable-cuda --ref foo --ds-collection macdebug --batch-size 32
"""
import json
import time

from torch import nn, optim

from ulfs import utils
from ulfs.stats import Stats
from ulfs.runner_base_v1 import RunnerBase

from ref_task import dataset_family_lib
from ref_task import params_groups
from ref_task.models import sender_model as sender_model_lib


class Runner(RunnerBase):
    def __init__(self):
        super().__init__(
            save_as_statedict_keys=['model', 'opt'],
            step_key='episode'
        )

    def setup(self, p):
        params_groups.process_args(args=p)

        print('')
        print('ref', p.ref)
        print('')

        if isinstance(p.cnn_sizes, str):
            p.cnn_sizes = [int(v) for v in p.cnn_sizes.split(',')]

        print('p', self.params)

        self.dataset = dataset_family_lib.build_family(**utils.filter_dict_by_prefix(p.__dict__, prefix='ds_'))
        self.ds_meta = self.dataset.meta
        print('ds_meta', json.dumps(self.ds_meta.__dict__, indent=2))

        self.vocab_size = len(self.dataset.i2w)
        print('vocab_size', self.vocab_size)
        self.grid_planes = self.ds_meta.grid_planes

        p.grid_size = self.ds_meta.grid_size
        p.max_colors = self.ds_meta.num_colors
        p.max_shapes = self.ds_meta.num_shapes
        self.in_seq_len = self.ds_meta.utt_len
        print('self.ds_meta.utt_len', self.ds_meta.utt_len)

        print('utt_len', self.ds_meta.utt_len)
        print('vocab_size', self.vocab_size)

        self.sender_model = sender_model_lib.build_sender_model(
            params=p, ds_meta=self.ds_meta, use_reinforce=False,
            utt_len=self.ds_meta.utt_len,
            vocab_size=self.vocab_size
        )
        if p.enable_cuda:
            self.sender_model = self.sender_model.cuda()
        print('sender_model', self.sender_model)

        self.opt = optim.Adam(lr=0.001, params=self.sender_model.parameters())
        self.crit = nn.CrossEntropyLoss()
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
            b = self.dataset.sample(batch_size=batch_size, split_name=split_name)
            input_examples_t, input_labels_t, receiver_examples_t, labels_t, hypotheses_t = map(b.__getitem__, [
                'inner_train_examples_t', 'inner_train_labels_t', 'inner_test_examples_t', 'inner_test_labels_t',
                'hypotheses_t'
            ])
            M, N = hypotheses_t.size()
            batch_size = N
            if p.enable_cuda:
                hypotheses_t = hypotheses_t.cuda()
                input_examples_t = input_examples_t.cuda()

            out = self.sender_model(images=input_examples_t, labels=input_labels_t)
            _, out_tokens = out.max(dim=-1)

            correct = (out_tokens == hypotheses_t).int()
            acc = correct.long().sum().item() / hypotheses_t.numel()
            out_flat = out.contiguous().view(self.ds_meta.utt_len * batch_size, self.vocab_size)
            hypotheses_t_flat = hypotheses_t.view(self.ds_meta.utt_len * batch_size)
            loss = self.crit(out_flat, hypotheses_t_flat)

            return out, out_tokens, correct, acc, loss

        self.sender_model.train()
        out, pred, correct, acc, loss = forward(split_name='train')

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

            format_str = ('e={episode} '
                          'eps={eps:.1f} '
                          'loss={loss:.3f} '
                          'acc={acc:.3f} ')
            for split_name in ['val_same', 'val_new', 'test_same', 'test_new']:
                eval_acc_l = []
                for it in range(1):
                    self.sender_model.eval()
                    out, pred, correct, eval_acc, eval_loss = forward(
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

    runner.add_param('--max-steps', type=int, default=0)
    runner.add_param('--max-mins', type=float, help='finish running if reach this elapsed time')

    params_groups.add_ds_args(runner)
    params_groups.add_common_args(runner)
    params_groups.add_sender_args(runner)
    params_groups.add_conv_args(runner)

    runner.parse_args()
    runner.setup_base()
    runner.run_base()
