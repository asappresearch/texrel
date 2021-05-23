# from ulfs.utils import expand


from ulfs import utils, runner_base_v1
from ref_task.datasets import all_datasets


def add_hypothesis_sampler_args(runner: runner_base_v1.RunnerBase):
    runner.add_param('--hypothesis-noise', type=float, default=0.0)
    runner.add_param('--hypothesis-tau', type=float, default=1.2)
    runner.add_param('--hypothesis-ent-reg', type=float, default=0.01)
    runner.add_param('--hypothesis-gumbel-soft', action='store_true')
    # runner.add_param('--hypothesis-sampler', type=str, default='HypothesisSamplerGumbel')
    # TODO, not duplicate the hypothesis sampler type and gumbel, reinforce
    runner.add_param('--hypothesis-exp-probs', action='store_true')


def add_receiver_args(runner: runner_base_v1.RunnerBase):
    runner.add_param('--multimodal-classifier', type=str, default='Cosine')
    runner.add_param(
        '--num-output-fcs', type=int, default=1,
        help='how many fully connected layers at the end of the multi modal classifier model')
    runner.add_param('--linguistic-encoder', type=str, default='RNN')
    runner.add_param('--linguistic-encoder-num-layers', type=int, default=1)
    runner.add_param('--linguistic-encoder-rnn-type', type=str, default='GRU')
    runner.add_param('--num-receiver-its', type=int, default=1)


def add_e2e_args(runner: runner_base_v1.RunnerBase):
    runner.add_param('--utt-len', type=int, default=10)
    runner.add_param('--vocab-size', type=int, default=21)
    # runner.add_param('--share-conv', action='store_true')
    runner.add_param('--max-steps', type=int, default=0)


def add_tre_args(runner: runner_base_v1.RunnerBase):
    runner.add_param('--tre-lr', type=float, default=0.01)
    runner.add_param('--tre-quiet', action='store_true')
    runner.add_param('--tre-steps', type=int, default=400)
    runner.add_param('--tre-max-samples', type=int, default=80)
    runner.add_param('--no-tre-zero-init', action='store_true')
    runner.add_param('--no-tre-bias', action='store_true')


def add_hypprop_args(runner):
    runner.add_param('--model-seed', type=int, default=123)

    runner.add_param('--train-receiver-by-row', action='store_true')

    runner.add_param('--obverter-trainer', type=str, default='BackpropM')
    runner.add_param('--sub-batch-size', type=int, default=16, help='for Backprop trainer')
    runner.add_param('--phase0-its', type=int, default=50)
    runner.add_param('--phase1-its', type=int, default=5)

    runner.add_param('--hyp-opt', type=str, default='Adam')
    runner.add_param('--hyp-checker-opt', type=str, default='Adam')
    runner.add_param('--meta-inner-opt', type=str, default='Adam', help='for reptile (maml always uses SGD')
    runner.add_param('--meta-outer-opt', type=str, default='Adam', help='for maml')
    runner.add_param('--meta-lr', type=float, default=0.001)
    runner.add_param('--meta-inner-steps', type=int, default=50)
    runner.add_param('--meta-accumulate-over', type=int, default=1)
    runner.add_param('--backpropm-accumulate-cols', type=int, default=1)
    runner.add_param('--no-backpropm-randomize-cols', action='store_true')

    runner.add_param('--hyp-l1-reg', type=float, default=0)
    runner.add_param('--hyp-hinge-reg', type=float, default=1)
    runner.add_param('--hyp-hinge-range', type=str, default='0,1')

    runner.add_param('--hyp-self-norm-l1-reg', type=float, default=0)
    runner.add_param('--hyp-self-norm-l2-reg', type=float, default=0)

    runner.add_param('--enable-predictor', action='store_true')
    # TODO add pred-activation back in
    runner.add_param('--pred-activation', type=str, help='[tanh|sigmoid]')

    runner.add_param('--hyp-first-symbol-margin', type=float)
    runner.add_param('--hyp-first-symbol-margin-reg', type=float, default=0)

    runner.add_param('--max-mfb-equiv', type=float, default=80)
    runner.add_param('--full-validation-every-mfb', type=float, default=0)

    runner.add_param('--swap-sender-receiver', action='store_true', help='for 2 agent')


def add_sender_args(runner: runner_base_v1.RunnerBase):
    runner.add_param('--image-seq-embedder', type=str, default='PrototypicalSender')
    runner.add_param('--no-sender-negex', action='store_true', help='disable negative examples')
    runner.add_param('--sender-decoder', type=str, default='RNNDecoder')
    runner.add_param('--sender-num-heads', type=int, default=1)
    runner.add_param('--sender-num-timesteps', type=int, default=5)
    runner.add_param('--sender-num-rnn-layers', type=int, default=1)


def add_conv_args(runner: runner_base_v1.RunnerBase):
    runner.add_param('--conv-preset', type=str, default='conv4', choices=['conv4', 'none'], help='optional')
    runner.add_param('--cnn-sizes', type=str, default='16,16')
    runner.add_param('--cnn-batch-norm', action='store_true')
    runner.add_param('--cnn-max-pooling-size', type=int, help='if None then no pooling, otherwise kernel size')
    runner.add_param('--preconv-model', type=str, default='StridedConv')
    runner.add_param('--preconv-stride', type=int, default=4)
    runner.add_param('--preconv-relu', action='store_true')
    runner.add_param('--preconv-dropout', type=float, default=0.2)
    runner.add_param('--preconv-embedding-size', type=int, default=16)


def add_common_args(runner: runner_base_v1.RunnerBase):
    runner.add_param('--embedding-size', type=int, default=128)
    runner.add_param('--dropout', type=float, default=0)
    runner.add_param('--lr', type=float, default=0.001)
    runner.add_param('--batch-size', type=int, default=128)
    runner.add_param('--repr', type=str, default='gumb', choices=['soft', 'gumb', 'discr'])


def add_ds_args(runner: runner_base_v1.RunnerBase):
    runner.add_param('--ds-family', type=str, choices=['texrel', 'shapeworld'], default='texrel')
    runner.add_param('--ds-shapeworld-folder', type=str, default='~/data/shapeworld')
    runner.add_param('--ds-filepath-templ', type=str, default='~/data/reftask/{ds_ref}.dat')
    runner.add_param('--ds-seed', type=int, default=123)
    runner.add_param('--ds-texture-size', type=int, default=4)
    runner.add_param('--ds-background-noise', type=float, default=0, help='std of noise (with mean 0.5)')
    runner.add_param('--ds-mean', type=float, default=0)
    runner.add_param('--ds-mean-std', type=float, default=0)

    runner.add_param('--ds-collection', type=str, default='128valsame')
    runner.add_param('--ds-tasks', type=str, default='Relations', help='tasks for training')
    runner.add_param('--ds-val-tasks', type=str,
                     help='tasks for validation and test (defaults to same as training)')
    runner.add_param('--ds-distractors', type=str, default='2')
    runner.add_param('--ds-val-distractors', type=str,
                     help='distractors for validation and test (defaults to same as training)')

    runner.add_param('--ds-refs', type=str)
    runner.add_param('--ds-val-refs', type=str)


def process_args(args):
    if 'ds_refs' in args.__dict__ and args.ds_refs is not None:
        args.ds_refs = args.ds_refs.split(',')
        if 'ds_val_refs' in args.__dict__ and args.ds_val_refs is not None:
            args.ds_val_refs = args.ds_val_refs.split(',')
        else:
            args.ds_val_refs = list(args.ds_refs)
    elif 'ds_collection' in args.__dict__:
        args.ds_tasks = args.ds_tasks.split(',')
        args.ds_distractors = [int(v) for v in args.ds_distractors.split(',')]
        if args.ds_val_tasks is not None:
            args.ds_val_tasks = args.ds_val_tasks.split(',')
        if args.ds_val_distractors is not None:
            args.ds_val_distractors = [int(v) for v in args.ds_val_distractors.split(',')]

        args.ds_collection = all_datasets.get(args.ds_collection, args.ds_collection)
        print(f'using ds_collection {args.ds_collection}')

    if 'hyp_hinge_range' in args.__dict__:
        hinge_range = [float(v.replace('min', '-')) for v in args.hyp_hinge_range.split(',')]
        args.hyp_hinge_range_start, args.hyp_hinge_range_end = hinge_range
        del args.__dict__['hyp_hinge_range']

    utils.reverse_args(args, 'hypothesis_exp_probs', 'hypothesis_apply_log')
    utils.reverse_args(args, 'hypothesis_gumbel_soft', 'hypothesis_gumbel_hard')

    if 'cnn_sizes' in args.__dict__ and isinstance(args.cnn_sizes, str):
        args.cnn_sizes = [int(v) for v in args.cnn_sizes.split(',')]

    args.hypothesis_sampler = {
        'soft': 'HypothesisSamplerGaussianNoise',
        'gumb': 'HypothesisSamplerGumbel',
        'discr': 'HypothesisSamplerREINFORCE'
    }[args.repr]
    del args.__dict__['repr']

    if getattr(args, 'conv_preset', 'none') != 'none':
        if args.conv_preset.lower() == 'conv4':
            print('using Conv4 convolutional network')
            args.cnn_sizes = [64, 64, 64, 64]
            args.cnn_batch_norm = True
            args.cnn_max_pooling_size = 2
            args.preconv_model = 'Identity'
        else:
            raise ValueError('conv_preset not recognized ' + args.conv_preset)

    args.device = 'cuda' if getattr(args, 'enable_cuda', False) else 'cpu'

    for k in list(args.__dict__.keys()):
        if k.startswith('no_'):
            pos_k = k.replace('no_', '', 1)
            setattr(args, pos_k, not getattr(args, k))
            del args.__dict__[k]
