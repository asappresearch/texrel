# texrel

Code for paper "TexRel: a Green Family of Datasets for Emergent Communications on Relations", Perkins, 2021, https://arxiv.org/abs/2105.12804

## Pre-requisites

- have python 3.7
- have run the following:
```
pip install -e .
pip install -r requirements.txt
```

## Download the data

- Download and untar https://public-dataset-model-store.awsdev.asapp.com/hp/papers/texrel/texrel.tar

## To create the datasets:

If running on a single machine:
```
python texrel/create_collection.py --ref macdebug --num-train 32 --num-val-same 32 --num-val-new 32 --num-test-same 32 --num-test-new 32
python texrel/create_collection.py --ref ords064sc9 --inner-train-pos 64 --inner-train-neg 64 --inner-test-pos 64 --inner-test-neg 64 --num-distractors 0,2 --num-colors 9 --num-shapes 9
python texrel/create_collection.py --ref ords055sc4 --inner-train-pos 64 --inner-train-neg 64 --inner-test-pos 64 --inner-test-neg 64 --num-distractors 2 --num-colors 4 --num-shapes 4
python texrel/create_collection.py --ref ords056sc3 --inner-train-pos 64 --inner-train-neg 64 --inner-test-pos 64 --inner-test-neg 64 --num-distractors 2 --num-colors 3 --num-shapes 3
python texrel/create_collection.py --ref ords057sc5 --inner-train-pos 64 --inner-train-neg 64 --inner-test-pos 64 --inner-test-neg 64 --num-distractors 2 --num-colors 5 --num-shapes 5
python texrel/create_collection.py --ref ords058sc6 --inner-train-pos 64 --inner-train-neg 64 --inner-test-pos 64 --inner-test-neg 64 --num-distractors 2 --num-colors 6 --num-shapes 6
python texrel/create_collection.py --ref ords059sc7 --inner-train-pos 64 --inner-train-neg 64 --inner-test-pos 64 --inner-test-neg 64 --num-distractors 2 --num-colors 7 --num-shapes 7
python texrel/create_collection.py --ref ords060sc8 --inner-train-pos 64 --inner-train-neg 64 --inner-test-pos 64 --inner-test-neg 64 --num-distractors 2 --num-colors 8 --num-shapes 8
python texrel/create_collection.py --ref ords061sc9 --inner-train-pos 64 --inner-train-neg 64 --inner-test-pos 64 --inner-test-neg 64 --num-distractors 2 --num-colors 9 --num-shapes 9
python texrel/create_collection.py --ref ords062 --inner-train-pos 64 --inner-train-neg 64 --inner-test-pos 64 --inner-test-neg 64 --num-distractors 2 --hypothesis-generators Relations --available-preps LeftOf
```

If you have access to a job submission system, that can distribute jobs over multiple machines, then, given you provide a script `ulfs_submit.py`, in the `PATH`, which can be called like `-r [reference] [... other args you can ignore] --  [script path] [script args]`, and which will run in each job `[script path] --ds-ref [reference] [script args]` then you can run instead:

```
python texrel/create_collection.py --ref macdebug --num-train 32 --num-val-same 32 --num-val-new 32 --num-test-same 32 --num-test-new 32
python texrel/create_collection_submit.py --ref ords064sc9 --inner-train-pos 64 --inner-train-neg 64 --inner-test-pos 64 --inner-test-neg 64 --num-distractors 0,2 --num-colors 9 --num-shapes 9
python texrel/create_collection_submit.py --ref ords055sc4 --inner-train-pos 64 --inner-train-neg 64 --inner-test-pos 64 --inner-test-neg 64 --num-distractors 2 --num-colors 4 --num-shapes 4
python texrel/create_collection_submit.py --ref ords056sc3 --inner-train-pos 64 --inner-train-neg 64 --inner-test-pos 64 --inner-test-neg 64 --num-distractors 2 --num-colors 3 --num-shapes 3
python texrel/create_collection_submit.py --ref ords057sc5 --inner-train-pos 64 --inner-train-neg 64 --inner-test-pos 64 --inner-test-neg 64 --num-distractors 2 --num-colors 5 --num-shapes 5
python texrel/create_collection_submit.py --ref ords058sc6 --inner-train-pos 64 --inner-train-neg 64 --inner-test-pos 64 --inner-test-neg 64 --num-distractors 2 --num-colors 6 --num-shapes 6
python texrel/create_collection_submit.py --ref ords059sc7 --inner-train-pos 64 --inner-train-neg 64 --inner-test-pos 64 --inner-test-neg 64 --num-distractors 2 --num-colors 7 --num-shapes 7
python texrel/create_collection_submit.py --ref ords060sc8 --inner-train-pos 64 --inner-train-neg 64 --inner-test-pos 64 --inner-test-neg 64 --num-distractors 2 --num-colors 8 --num-shapes 8
python texrel/create_collection_submit.py --ref ords061sc9 --inner-train-pos 64 --inner-train-neg 64 --inner-test-pos 64 --inner-test-neg 64 --num-distractors 2 --num-colors 9 --num-shapes 9
python texrel/create_collection_submit.py --ref ords062 --inner-train-pos 64 --inner-train-neg 64 --inner-test-pos 64 --inner-test-neg 64 --num-distractors 2 --hypothesis-generators Relations --available-preps LeftOf
```

Each call to `create_collection_submit.py` will launch one job for each of the 10 task types, i.e. Colors1, Colors2, Shapes1, etc. Note that for bests results you should provide also a script `ulfs_logs.sh` in the PATH which can be called like `ulfs_logs.sh [reference]`, and which will output the `stdout` of the launched job corresponding to reference `[reference]`. However, if you do not supply this, this will not prevent the jobs being launched.

### Dataset md5sums

`md5sum`s for each datafile can be found at [md5sums.txt](md5sums.txt)

## To use the datasets in your code

- you will need to import `texrel.dataset_runtime`, and create an instance of `texrel.dataset_runtime.TexRelDataset`

## To run the experiments from the paper

### Comparison with Shapeworld

Pre-requisites:
- clone the `emergent-communications` branch of the LSL repo fork at [https://github.com/asappresearch/lsl/tree/emergent-communications](https://github.com/asappresearch/lsl/tree/emergent-communications), from the parent folder of the `texrel` repo. That is after the fork, should have a folder containing both this `texrel` repo, and the forked `lsl` repo. Then from within the `texrel` folder do:

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
python ref_task/analysis/texrel/reduce_send_recv.py --out-ref trsr005 --in-refs trs009 trs010 trs011 trs012 --direction send
python ref_task/analysis/texrel/reduce_send_recv.py --direction recv --in-refs trr010 trr011 trr012 trr013 trr014 --out-ref trs006
python ref_task/analysis/texrel/reduce_e2e.py --out-ref trred015 --in-refs tree043 tree044 tree045 tree046 tree047
python ref_task/analysis/texrel/reduce_multitask.py --out-ref trred014 --in-refs trmt018 trmt019 trmt020 trmt021 trmt021
python ref_task/analysis/texrel/reduce_varatts.py --in-csvs traa00[7-9].csv traa01[0-1].csv --out-csv-long varatts_long.csv --out-csv-wide varatts_wide.csv
# following is not actually a reduction, just adds `gen_err` and `ptre` columns:
python ref_task/analysis/texrel/section7_comp_add_gen_err.py --in-csv trmc015.csv --out-csv trmc015-for-latex.csv  
```

## To run unit-tests

```
pytest -v .
```

## LSL repo fork

The code changes to LSL, in order to enable use the study of emergent communications using it, will be made available at [https://github.com/asappresearch/lsl/tree/emergent-communications](https://github.com/asappresearch/lsl/tree/emergent-communications). To run this code, please see the section above. Or else, to run standalone, you can do e.g.:

```
git clone git@github.com:asappresearch/lsl.git -b emergent-communications
cd lsl/shapeworld
python lsl/train.py --cuda --infer_hyp --hypo_lambda 1.0 --batch_size 100 --seed 123 --e2e-emergent-communications --data_dir /gfs/nlp/hp/data exp/l3 --backbone conv4 --epochs 50
```

## Citing

If you find the TexRel code useful, please cite:

```
@misc{perkins2021texrel,
      title={TexRel: a Green Family of Datasets for Emergent Communications on Relations}, 
      author={Hugh Perkins},
      year={2021},
      eprint={2105.12804},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
