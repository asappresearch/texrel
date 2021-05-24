"""
just some dummy model to demo using RunnerBase

run like:

python tests/dummy_model.py --ref fooa --disable-cuda --render-every-seconds 0
"""
import torch
from torch import nn, optim
import numpy as np

from ulfs.stats import Stats
from ulfs.runner_base_v1 import RunnerBase


class Model(nn.Module):
    def __init__(self, input_features, embedding_size, num_classes):
        super().__init__()
        # self.embedding_size = embedding_size
        # self.num_classes = num_classes
        self.h1 = nn.Linear(input_features, embedding_size)
        self.h2 = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        x = self.h1(x)
        x = torch.tanh(x)
        x = self.h2(x)
        return x


class Runner(RunnerBase):
    def __init__(self):
        super().__init__(
            save_as_statedict_keys=['model', 'opt'],
            additional_save_keys=[],
            step_key='episode'
        )

    def setup(self, p):
        print('setup start')
        torch.manual_seed(p.seed)
        np.random.seed(p.seed)
        print('seeded')

        self.model = Model(input_features=p.input_features, embedding_size=p.embedding_size, num_classes=p.num_classes)
        if p.enable_cuda:
            self.model = self.model.cuda()
        self.opt = optim.Adam(lr=0.001, params=self.model.parameters())
        self.crit = nn.CrossEntropyLoss()

        self.N = p.batch_size * 50
        self.data = torch.rand(self.N, p.input_features)
        self.labels = torch.from_numpy(np.random.choice(p.num_classes, self.N))

        self.stats = Stats([
            'episodes_count',
            'loss_sum'
        ])
        print('setup end')

    def step(self, p):
        render = self.should_render()

        num_batches = self.N // p.batch_size
        perm_idx = torch.from_numpy(np.random.choice(self.N, self.N, replace=False))
        for b in range(num_batches):
            b_idxes = perm_idx[b * p.batch_size: (b + 1) * p.batch_size]
            # in_b = self.data[b * p.batch_size: (b + 1) * p.batch_size]
            # out_b = self.labels[b * p.batch_size: (b + 1) * p.batch_size]
            in_b = self.data[b_idxes]
            out_b = self.labels[b_idxes]
            output = self.model(in_b)
            loss = self.crit(output, out_b)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            self.stats.loss_sum += loss.item()
            self.stats.episodes_count += 1
        # self.stats.loss_sum += loss.item()

        if render:
            stats = self.stats
            log_dict = {
                'loss': stats.loss_sum / stats.episodes_count,
            }
            self.print_and_log(
                log_dict,
                formatstr='e={episode} '
                          'loss {loss:.5f} '
            )
            stats.reset()


if __name__ == '__main__':
    runner = Runner()
    runner.add_param('--seed', type=int, default=1)
    runner.add_param('--input-features', type=int, default=12)
    runner.add_param('--embedding-size', type=int, default=20)
    # runner.add_param('--output-size', type=int, default=1)
    runner.add_param('--num-classes', type=int, default=100)
    runner.add_param('--batch-size', type=int, default=128)
    runner.parse_args()
    runner.setup_base()
    runner.run_base()
