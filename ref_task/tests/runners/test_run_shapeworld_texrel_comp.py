from ref_task.runners import run_shapeworld_texrel_comp


def test_lsl_rows_to_metrics():
    example_output = """
====>      (train)      Epoch:   1      Accuracy: 0.6106
====>        (val)      Epoch:   1      Accuracy: 0.5120
====>       (test)      Epoch:   1      Accuracy: 0.4998
====>   (val_same)      Epoch:   1      Accuracy: 0.5280
====>  (test_same)      Epoch:   1      Accuracy: 0.4995
====>      (train)      Epoch:   2      Loss: 64.3182
mean accuracy during training 0.625
====>      (train)      Epoch:   2      Accuracy: 0.6446
====>        (val)      Epoch:   2      Accuracy: 0.4960
====>       (test)      Epoch:   2      Accuracy: 0.5152
====>   (val_same)      Epoch:   2      Accuracy: 0.5100
====>  (test_same)      Epoch:   2      Accuracy: 0.4903
====>      (train)      Epoch:   3      Loss: 60.8819
mean accuracy during training 0.666
====>      (train)      Epoch:   3      Accuracy: 0.6756
====>        (val)      Epoch:   3      Accuracy: 0.5000
====>       (test)      Epoch:   3      Accuracy: 0.5022
====>   (val_same)      Epoch:   3      Accuracy: 0.5120
====>  (test_same)      Epoch:   3      Accuracy: 0.4825
====>      (train)      Epoch:   5      Loss: 54.7472
mean accuracy during training 0.722
====>      (train)      Epoch:   5      Accuracy: 0.7463
====>        (val)      Epoch:   5      Accuracy: 0.5200
====>       (test)      Epoch:   5      Accuracy: 0.5032
====>   (val_same)      Epoch:   5      Accuracy: 0.5180
====>  (test_same)      Epoch:   5      Accuracy: 0.4913
====>      (train)      Epoch:   6      Loss: 52.8679
mean accuracy during training 0.735
====>      (train)      Epoch:   6      Accuracy: 0.7446
====>        (val)      Epoch:   6      Accuracy: 0.5080
====>       (test)      Epoch:   6      Accuracy: 0.4923
====>   (val_same)      Epoch:   6      Accuracy: 0.5100
====>  (test_same)      Epoch:   6      Accuracy: 0.4930
====>      (train)      Epoch:   7      Loss: 49.7076
mean accuracy during training 0.760
====>      (train)      Epoch:   7      Accuracy: 0.7631
====>        (val)      Epoch:   7      Accuracy: 0.5500
====>       (test)      Epoch:   7      Accuracy: 0.5012
====>   (val_same)      Epoch:   7      Accuracy: 0.5060
====>  (test_same)      Epoch:   7      Accuracy: 0.4938
====>      (train)      Epoch:   8      Loss: 48.3630
mean accuracy during training 0.767
====>      (train)      Epoch:   8      Accuracy: 0.7714
====>        (val)      Epoch:   8      Accuracy: 0.5500
====>       (test)      Epoch:   8      Accuracy: 0.5160
====>   (val_same)      Epoch:   8      Accuracy: 0.5300
====>  (test_same)      Epoch:   8      Accuracy: 0.4870
====>      (train)      Epoch:   9      Loss: 46.0953
mean accuracy during training 0.778
====>      (train)      Epoch:   9      Accuracy: 0.7903
====>        (val)      Epoch:   9      Accuracy: 0.5260
====>       (test)      Epoch:   9      Accuracy: 0.5078
====>   (val_same)      Epoch:   9      Accuracy: 0.4940
====>  (test_same)      Epoch:   9      Accuracy: 0.5022
====>      (train)	Epoch:  50	Loss: 69.3388
mean accuracy during training 0.508
====>      (train)	Epoch:  50	Accuracy: 0.5063
====>        (val)	Epoch:  50	Accuracy: 0.4920
====>       (test)	Epoch:  50	Accuracy: 0.5040
====>   (val_same)	Epoch:  50	Accuracy: 0.5040
====>  (test_same)	Epoch:  50	Accuracy: 0.5032
====>      (train)	Epoch:  50	Loss: 69.3388
mean accuracy during training 0.508
====>      (train)	Epoch:  50	Accuracy: 0.5063
====>        (val)	Epoch:  50	Accuracy: 0.4920
====>       (test)	Epoch:  50	Accuracy: 0.5040
====>   (val_same)	Epoch:  50	Accuracy: 0.5040
====>  (test_same)	Epoch:  50	Accuracy: 0.5032
====> DONE
====> BEST EPOCH: 3
====>        (best_val)	Epoch: 3	Accuracy: 0.5080
====>   (best_val_same)	Epoch: 3	Accuracy: 0.5200
====>       (best_test)	Epoch: 3	Accuracy: 0.5060
====>  (best_test_same)	Epoch: 3	Accuracy: 0.5048
====>
====>    (best_val_avg)	Epoch: 3	Accuracy: 0.5140
====>   (best_test_avg)	Epoch: 3	Accuracy: 0.5054
"""
    metrics = run_shapeworld_texrel_comp.lsl_rows_to_metrics(example_output.split('\n'))
    print('metrics', metrics)
    assert metrics['best_val'] == 0.5080
    assert metrics['best_test'] == 0.5060
    assert metrics['best_test_avg'] == 0.5054
    assert metrics['best_epoch'] == 3
    assert metrics['total_epochs'] == 50
    assert metrics['train_acc'] == 0.6756
