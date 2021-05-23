"""
1. combine sender and receiver models together
2. ???

Hmmm, but, for now, let's backprop end-to-end, as reals. (ie no REINFORCE yet)

note to self: example debug command for dev:
python ref_task/run_end_to_end.py --ref foo --disable-cuda --ds-collection debugmac \
    --render-every-seconds 1 --max-steps 10 --batch-size 8
"""
import time
import random
import itertools
import os
from os import path
from typing import Tuple, Dict

import torch
from torch import nn, optim
import numpy as np

from ulfs.stats import Stats
from ulfs.runner_base_v1 import RunnerBase

from ref_task import dataset_family_lib
from ulfs import utils, tensor_utils, metrics as metrics_lib, tre as tre_lib, nn_utils
from ref_task import params_groups
from ref_task.models import sender_model as sender_model_lib, samplers, receiver_model as receiver_model_lib


class EndToEndModel(nn.Module):
    def __init__(
            self,
            sender_model: sender_model_lib.SenderModel,
            receiver_model: receiver_model_lib.ReceiverModel,
            sampler: samplers.Sampler) -> None:
        super().__init__()
        self.sender_model = sender_model
        self.receiver_model = receiver_model
        self.sampler = sampler

    def forward(
            self,
            input_images: torch.Tensor,
            input_labels: torch.Tensor,
            receiver_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # sender
        utt_logits = self.sender_model(images=input_images, labels=input_labels)

        utt_probs = self.sampler(utt_logits)
        utts_size = list(utt_probs.size())
        utt_probs_expanded = utt_probs.unsqueeze(1).expand(
            utts_size[0], receiver_images.size(0), *utts_size[1:])
        merger = tensor_utils.DimMerger()
        utts_flat = merger.merge(utt_probs_expanded.contiguous(), 1, 2)
        recv_images_flat = merger.merge(receiver_images, 0, 1)

        # receiver
        out_logits_flat = self.receiver_model(
            utts=utts_flat, images=recv_images_flat)
        out_logits = merger.resplit(out_logits_flat, 0)

        return utt_logits, out_logits


class Runner(RunnerBase):
    def __init__(self):
        super().__init__(
            save_as_statedict_keys=['model', 'opt'],
            step_key='episode'
        )

    def setup(self, p):
        params_groups.process_args(args=p)

        self.dataset = dataset_family_lib.build_family(**utils.filter_dict_by_prefix(p.__dict__, prefix='ds_'))
        self.ds_meta = self.dataset.meta

        torch.manual_seed(p.seed)
        np.random.seed(p.seed)
        random.seed(p.seed)

        self.sender_model = sender_model_lib.build_sender_model(
            params=p, ds_meta=self.ds_meta, use_reinforce=False, pre_conv=None,
            utt_len=p.utt_len, vocab_size=p.vocab_size)

        self.receiver_model = receiver_model_lib.build_receiver_model(
            params=p, ds_meta=self.ds_meta,
            vocab_size=p.vocab_size, utt_len=p.utt_len,
            pre_conv=self.sender_model.pre_conv
        )
        self.sampler = samplers.build_sampler(**utils.filter_dict_by_prefix(p.__dict__, prefix='sampler_'))

        self.model = EndToEndModel(
            sampler=self.sampler,
            sender_model=self.sender_model,
            receiver_model=self.receiver_model)
        if p.enable_cuda:
            self.model = self.model.cuda()
        print('model', self.model)
        num_params = nn_utils.get_num_parameters(self.model)
        print('num_params', num_params)
        nn_utils.dump_parameter_sizes(self.model)

        Opt = getattr(optim, p.e2e_opt)
        self.opt = Opt(lr=0.001, params=self.model.parameters())
        self.crit = nn.CrossEntropyLoss()
        self.stats = Stats([
            'episodes_count',
            'loss_sum',
            'acc_sum'
        ])
        self.last_val_infos = []

    def step(self, p):
        episode = self.episode
        render = self.should_render()
        device = p.device

        def forward(batch: Dict[str, torch.Tensor]):
            train_examples_t = batch['inner_train_examples_t'].to(device)
            train_labels_t = batch['inner_train_labels_t'].long().to(device)
            test_examples_t = batch['inner_test_examples_t'].to(device)
            test_labels_t = batch['inner_test_labels_t'].long().to(device)

            utt_logits, out_logits = self.model(
                input_labels=train_labels_t,
                input_images=train_examples_t, receiver_images=test_examples_t)
            _, pred = out_logits.max(dim=-1)
            correct = pred.view(-1) == test_labels_t.view(-1)
            acc = correct.long().sum().item() / p.batch_size / test_examples_t.size(0)
            loss = self.crit(
                out_logits.view(-1, out_logits.size(-1)), test_labels_t.view(-1))

            _, utt_tokens = utt_logits.detach().max(dim=-1)
            return out_logits, utt_tokens, correct, acc, loss, batch

        self.model.train()
        batch = self.dataset.sample(batch_size=p.batch_size, split_name='train')
        out_logits, _, correct, acc, loss, _ = forward(batch=batch)

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

        if p.max_mins is not None and p.max_mins > 0 and time.time() - self.start_time >= p.max_mins * 60:
            print('reached terminate time => terminating')
            self.finish = True
            terminate_reason = 'timeout'

        if render or self.finish:
            stats = self.stats
            log_dict = {
                'loss': stats.loss_sum / stats.episodes_count,
                'acc': stats.acc_sum / stats.episodes_count,
                'this_render_train_time': time.time() - self.last_print,
            }
            eval_start = time.time()

            def run_eval(split_name: str) -> Tuple[str, Dict[str, float]]:
                """
                run evaluation, i.e. with dropout etc set to False,
                but we might be using the training dataset, or the holdout dataset,
                depending on set_name

                Parameters
                ----------
                split_name: str
                    name of the dataset to use, choose from 'train' vs 'holdout'

                Returns
                -------
                Tuple[str, Dict[str, float]]
                    formatstr, metrics
                """
                metrics: Dict[str, float] = {}
                acc_l = []
                tokens_l = []
                hypotheses_english_l = []
                hypotheses_l = []
                hypothesis_structures_l = []
                self.model.eval()
                # for it in range(10):
                for batch in self.dataset.holdout_iterator(batch_size=p.batch_size, split_name=split_name):
                    # batch = self.dataset.sample(batch_size=p.batch_size, split_name=split_name)
                    with torch.no_grad():
                        _, utt_tokens, _, eval_acc, _, batch = forward(batch=batch)
                    acc_l.append(eval_acc)
                    tokens_l.append(utt_tokens.detach())
                    hypotheses_l.append(batch['hypotheses_t'])
                    if 'hypotheses_structured' in batch:
                        hypothesis_structures_l.append(batch['hypotheses_structured'])
                    hypotheses_english_l += batch['hypotheses_english']
                metrics[f'{split_name}_acc'] = sum(acc_l) / len(acc_l)
                hypothesis_structures_l = list(itertools.chain.from_iterable(hypothesis_structures_l))
                tokens_t = torch.cat(tokens_l, dim=-1).transpose(0, 1)
                hypotheses_t = torch.cat(hypotheses_l, dim=-1).transpose(0, 1)

                if p.tensor_dumps_templ_path:
                    out_dict = {
                        'hypotheses_structures': hypothesis_structures_l,
                        'tokens_t': tokens_t,
                        'hypotheses_t': hypotheses_t,
                        'vocab_size': p.vocab_size,
                        'utt_len': p.utt_len,
                        'hypotheses_english': hypotheses_english_l,
                    }
                    out_path = p.tensor_dumps_templ_path.format(batch=episode, split_name=split_name)
                    out_folder = path.dirname(out_path)
                    if not path.exists(out_folder):
                        os.makedirs(out_folder)
                    torch.save(out_dict, out_path)
                    print('wrote ' + out_path)

                if self.finish and p.tensor_dumps_templ_path_on_terminate:
                    out_dict = {
                        'hypotheses_structures': hypothesis_structures_l,
                        'tokens_t': tokens_t,
                        'hypotheses_t': hypotheses_t,
                        'vocab_size': p.vocab_size,
                        'utt_len': p.utt_len,
                        'hypotheses_english': hypotheses_english_l,
                    }
                    out_path = p.tensor_dumps_templ_path_on_terminate.format(split_name=split_name)
                    out_folder = path.dirname(out_path)
                    if not path.exists(out_folder):
                        os.makedirs(out_folder)
                    torch.save(out_dict, out_path)
                    print('wrote ' + out_path)

                # precision, recall
                before_prec_rec = time.time()
                ground_clustering = metrics_lib.cluster_strings(hypotheses_english_l)
                pred_clustering = metrics_lib.cluster_utts(tokens_t.transpose(0, 1))
                prec, rec = metrics_lib.calc_cluster_prec_recall(
                    ground=ground_clustering, pred=pred_clustering)
                prec_rec_time = time.time() - before_prec_rec
                metrics[f'{split_name}_prec'] = prec
                metrics[f'{split_name}_rec'] = rec
                metrics[f'{split_name}_gnd_clusters'] = ground_clustering.max().item() + 1
                metrics[f'{split_name}_pred_clusters'] = pred_clustering.max().item() + 1
                metrics[f'{split_name}_prec_rec_time'] = prec_rec_time

                # calculate rho
                before_rho = time.time()
                rho = metrics_lib.topographic_similarity(
                    utts=tokens_t, labels=hypotheses_t
                )
                rho_time = time.time() - before_rho
                metrics[f'{split_name}_rho'] = rho
                metrics[f'{split_name}_rho_time'] = rho_time

                formatstr = (
                    f'{split_name}['
                    f'acc={{{split_name}_acc:.3f}} '
                    f'rho={{{split_name}_rho:.3f}} '
                    f'prec={{{split_name}_prec:.3f}} '
                    f'rec={{{split_name}_rec:.3f}} '
                    f'gnd_cls={{{split_name}_gnd_clusters:.0f}} '
                    f'pred_cls={{{split_name}_pred_clusters:.0f}} '
                )

                # calculate tre
                if p.evaluate_tre or (p.evaluate_tre_on_terminate and self.finish):
                    print('episode', episode, split_name, 'calc tre')
                    before_tre = time.time()
                    tokens_onehot = tensor_utils.idxes_to_onehot(idxes=tokens_t, vocab_size=p.vocab_size)
                    tokens_onehot = tensor_utils.merge_dims(tokens_onehot, -2, -1)
                    comp_fn = tre_lib.ProjectionSumCompose(
                        num_terms=2,
                        vocab_size=p.vocab_size,
                        msg_len=p.utt_len,
                        bias=p.tre_bias,
                    )
                    tre = tre_lib.evaluate(
                        reps=tokens_onehot,
                        oracle_structures=hypothesis_structures_l,
                        comp_fn=comp_fn,
                        distance_fn=tre_lib.L1Dist(),
                        tre_lr=p.tre_lr,
                        quiet=p.tre_quiet,
                        steps=p.tre_steps,
                        max_samples=p.tre_max_samples,
                        zero_init=p.tre_zero_init)
                    tre_time = time.time() - before_tre
                    metrics[f'{split_name}_tre'] = tre
                    metrics[f'{split_name}_tre_time'] = tre_time

                    formatstr += (
                        f'tre={{{split_name}_tre:.3f}} '
                    )
                formatstr = formatstr.rstrip() + ']'

                return formatstr, metrics
            formatstr = (
                'e={episode} '
                'loss={loss:.3f} '
                'acc={acc:.3f} '
                'rel_eval_time={this_render_eval_rel_time:.3f} '
            )
            holdout_formatstr, holdout_metrics = run_eval(
                split_name='val_same')
            log_dict.update(holdout_metrics)
            formatstr += holdout_formatstr

            holdout_formatstr, holdout_metrics = run_eval(
                split_name='val_new')
            log_dict.update(holdout_metrics)
            formatstr += holdout_formatstr

            log_dict['this_render_eval_time'] = time.time() - eval_start
            log_dict['this_render_eval_rel_time'] = (time.time() - eval_start) / (eval_start - self.last_print)
            log_dict['batch'] = episode

            if p.early_stop_metric is not None:
                early_stop_metric = log_dict[p.early_stop_metric]
                val_info = {
                    'step': episode,
                    'early_stop_score': early_stop_metric,
                    'log_dict': dict(log_dict),
                    'state_dict': self.model.state_dict()
                }
                self.last_val_infos.append(val_info)
                self.last_val_infos = self.last_val_infos[-p.early_stop_patience:]
                if len(self.last_val_infos) >= p.early_stop_patience:
                    early_stop_scores = [info['early_stop_score'] for info in self.last_val_infos]
                    oldest_metric_score = early_stop_scores[0]
                    early_stopping = False
                    if p.early_stop_direction == 'max':
                        if oldest_metric_score >= max(early_stop_scores):
                            print('self.val_scores[-p.early_stop_patience:]', early_stop_scores)
                            print('early stopping')
                            early_stopping = True
                    else:
                        if oldest_metric_score <= min(early_stop_scores):
                            print('self.val_scores[-p.early_stop_patience:]', early_stop_scores)
                            print('early stopping')
                            early_stopping = True
                    if early_stopping:
                        terminate_reason = 'early stopping'
                        self.finish = True
                        best_val_info = self.last_val_infos[0]
                        log_dict = dict(best_val_info['log_dict'])
                        self.model.load_state_dict(best_val_info['state_dict'])
                        print('loaded model from best episode', best_val_info['step'])

            if self.finish:
                holdout_formatstr, holdout_metrics = run_eval(
                    split_name='test_same')
                log_dict.update(holdout_metrics)
                formatstr += holdout_formatstr

                holdout_formatstr, holdout_metrics = run_eval(
                    split_name='test_new')
                log_dict.update(holdout_metrics)
                formatstr += holdout_formatstr

            self.print_and_log(log_dict, formatstr=formatstr)
            stats.reset()

        if self.finish:
            print('log_dict', log_dict)
            self.res = dict(log_dict)
            self.res.update({
                'total_elapsed': (time.time() - self.start_time),
                'terminate_reason': terminate_reason,
            })
            print('self.res', self.res)


def add_e2e_args(runner):
    runner.add_param('--sampler-model', type=str, default='Softmax')
    runner.add_param('--no-sampler-hard', action='store_true')
    runner.add_param('--sampler-tau', type=float, default=1.2)
    runner.add_param('--sampler-gaussian-noise', type=float, default=0.0)
    runner.add_param('--max-mins', type=float, help='finish running if reach this elapsed time')

    runner.add_param('--early-stop-patience', type=int, default=10,
                     help='how many validations to run before early stopping')
    runner.add_param('--early-stop-metric', type=str)
    runner.add_param('--early-stop-direction', type=str, default='max')

    runner.add_param('--e2e-opt', type=str, default='Adam')

    runner.add_param('--seed', type=int, default=123)

    runner.add_param('--tensor-dumps-templ-path', type=str,
                     help='if not provided, then no dumps, use {batch} to represent batch number'
                     ' and {split_name} for split name')
    runner.add_param('--tensor-dumps-templ-path-on-terminate', type=str,
                     help='if not provided, then no dumps, use'
                     ' {split_name} for split name')
    runner.add_param('--evaluate-tre', action='store_true')
    runner.add_param('--evaluate-tre-on-terminate', action='store_true', help='only when finish is True')


if __name__ == '__main__':
    runner = Runner()

    add_e2e_args(runner)

    params_groups.add_ds_args(runner)
    params_groups.add_e2e_args(runner)
    params_groups.add_tre_args(runner)
    params_groups.add_conv_args(runner)
    params_groups.add_common_args(runner)
    params_groups.add_sender_args(runner)
    params_groups.add_receiver_args(runner)

    runner.parse_args()
    runner.setup_base()
    runner.run_base()
