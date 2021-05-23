# texrel
texrel

## Pre-requisites

- have python 3.7
- have run the following:
```
pip install -e .
pip install -r requirements.txt
```

## To create the datasets:


## To use the datasets in your code

- you will need to import `texrel.dataset_runtime`, and create an instance of `texrel.dataset_runtime.TexRelDataset`

## To run the experiments from the paper

### Comparison with Shapeworld

```
python ref_task/runners/run_shapeworld_texrel_comp.py --ref trbase043 --early-stop-metric val_same_acc --batch-size 32 --seed-base 123
python ref_task/runners/run_shapeworld_texrel_comp.py --ref trbase044 --early-stop-metric val_same_acc --batch-size 32 --seed-base 124
python ref_task/runners/run_shapeworld_texrel_comp.py --ref trbase045 --early-stop-metric val_same_acc --batch-size 32 --seed-base 125
python ref_task/runners/run_shapeworld_texrel_comp.py --ref trbase046 --early-stop-metric val_same_acc --batch-size 32 --seed-base 126
python ref_task/runners/run_shapeworld_texrel_comp.py --ref trbase047 --early-stop-metric val_same_acc --batch-size 32 --seed-base 127
```

### Comparison of sender architectures

```
python ref_task/runners/run_arch_send_comparison.py --ref trs009 --max-mins 5 --batch-size 32 --ds-collection ords064sc9 --seed 123
python ref_task/runners/run_arch_send_comparison.py --ref trs010 --max-mins 5 --batch-size 32 --ds-collection ords064sc9 --seed 124
python ref_task/runners/run_arch_send_comparison.py --ref trs011 --max-mins 5 --batch-size 32 --ds-collection ords064sc9 --seed 125
python ref_task/runners/run_arch_send_comparison.py --ref trs012 --max-mins 5 --batch-size 32 --ds-collection ords064sc9 --seed 126
python ref_task/runners/run_arch_send_comparison.py --ref trs013 --max-mins 5 --batch-size 32 --ds-collection ords064sc9 --seed 127
```

### Comparison of receiver architectures

```
python ref_task/runners/run_arch_recv_comparison.py --ref trr010 --max-mins 5 --batch-size 32 --ds-collection ords064sc9 --seed 123
python ref_task/runners/run_arch_recv_comparison.py --ref trr011 --max-mins 5 --batch-size 32 --ds-collection ords064sc9 --seed 124
python ref_task/runners/run_arch_recv_comparison.py --ref trr012 --max-mins 5 --batch-size 32 --ds-collection ords064sc9 --seed 125
python ref_task/runners/run_arch_recv_comparison.py --ref trr013 --max-mins 5 --batch-size 32 --ds-collection ords064sc9 --seed 126
python ref_task/runners/run_arch_recv_comparison.py --ref trr014 --max-mins 5 --batch-size 32 --ds-collection ords064sc9 --seed 127
```

### Comparison of end-to-end architectures

```
python ref_task/runners/run_e2e_comparison.py --ref tree043 --early-stop-metric val_same_acc --batch-size 32 --sampler-model Gumbel --max-mins -1
python ref_task/runners/run_e2e_comparison.py --ref tree044 --early-stop-metric val_same_acc --batch-size 32 --sampler-model Gumbel --max-mins -1 --seed-base 124
python ref_task/runners/run_e2e_comparison.py --ref tree045 --early-stop-metric val_same_acc --batch-size 32 --sampler-model Gumbel --max-mins -1 --seed-base 125
python ref_task/runners/run_e2e_comparison.py --ref tree046 --early-stop-metric val_same_acc --batch-size 32 --sampler-model Gumbel --max-mins -1 --seed-base 126
python ref_task/runners/run_e2e_comparison.py --ref tree047 --early-stop-metric val_same_acc --batch-size 32 --sampler-model Gumbel --max-mins -1 --seed-base 127
```

### Effect of multi-task training

```
python ref_task/runners/run_multitask_learning.py --ref trmt018 --batch-size 32 --sampler-model Gumbel --early-stop-metric val_same_acc --render-every-seconds -1 --render-every-steps 300 --seed-base 123
python ref_task/runners/run_multitask_learning.py --ref trmt019 --batch-size 32 --sampler-model Gumbel --early-stop-metric val_same_acc --render-every-seconds -1 --render-every-steps 300 --seed-base 124
python ref_task/runners/run_multitask_learning.py --ref trmt020 --batch-size 32 --sampler-model Gumbel --early-stop-metric val_same_acc --render-every-seconds -1 --render-every-steps 300 --seed-base 125
python ref_task/runners/run_multitask_learning.py --ref trmt021 --batch-size 32 --sampler-model Gumbel --early-stop-metric val_same_acc --render-every-seconds -1 --render-every-steps 300 --seed-base 126
python ref_task/runners/run_multitask_learning.py --ref trmt022 --batch-size 32 --sampler-model Gumbel --early-stop-metric val_same_acc --render-every-seconds -1 --render-every-steps 300 --seed-base 127
```


### Effect of number of attributes, and number of attribute values

```
python ref_task/runners/run_numatts_numvalues.py --ref traa007 --sampler-model Gumbel --batch-size 32 --max-steps 5000 --seed 123
python ref_task/runners/run_numatts_numvalues.py --ref traa008 --sampler-model Gumbel --batch-size 32 --max-steps 5000 --seed 124
python ref_task/runners/run_numatts_numvalues.py --ref traa009 --sampler-model Gumbel --batch-size 32 --max-steps 5000 --seed 125
python ref_task/runners/run_numatts_numvalues.py --ref traa010 --sampler-model Gumbel --batch-size 32 --max-steps 5000 --seed 126
python ref_task/runners/run_numatts_numvalues.py --ref traa011 --sampler-model Gumbel --batch-size 32 --max-steps 5000 --seed 127
```

### Reproducing section 7 of TRE paper

```
python ref_task/runners/run_measuring_comp_section7.py --ref trmc015 --max-steps 3000 --ds-collection 128leftonly --utt-len 5 --vocab-size 26 --batch-size 32 --sampler-model Gumbel
```

## To reduce the results over the seeds, generate results tables, etc

```
python ref_task/analysis/texrel/reduce_vs_shapeworld.py --out-ref trred013 --in-refs trbase038 trbase039 trbase040 trbase041 trbase042
python ref_task/analysis/reduce_send_recv.py --out-ref trsr005 --in-refs trs009 trs010 trs011 trs012 --direction send
python ref_task/analysis/reduce_send_recv.py --direction recv --in-refs trr010 trr011 trr012 trr013 trr014 --out-ref trs006
python ref_task/analysis/texrel/reduce_e2e.py --out-ref trred015 --in-refs tree043 tree044 tree045 tree046 tree047
python ref_task/analysis/texrel/reduce_multitask.py --out-ref trred014 --in-refs trmt018 trmt019 trmt020 trmt021 trmt021
# todo: put reduce for num attributes and values here
# todo: put reduce for section 7 reproduction here
```
