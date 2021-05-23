import pytest
from texrel.dataset_runtime import TexRelDataset


@pytest.mark.skip()
def test_test_iter():
    ds = TexRelDataset(
        ds_refs=['test_ts7_1', 'test_ts7_2'],
        ds_filepath_templ='~/data/reftask/{ds_ref}.dat',
        ds_seed=123,
        ds_texture_size=4,
        ds_background_noise=0,
        ds_mean=0,
        ds_mean_std=0
    )
    iterator = ds.holdout_iterator(batch_size=4)
    for b, batch in enumerate(iterator):
        print(b, batch['N'], batch.keys())
    for k, v in batch.items():
        if isinstance(v, int):
            print(k, v)
        else:
            print(k, v.size())
    print('ds.meta', ds.meta)


@pytest.mark.skip()
def test_hypotheses_single_ds_rels():
    ds = TexRelDataset(
        ds_refs=['ds12_rels'],
        ds_filepath_templ='~/data/reftask/{ds_ref}.dat',
        ds_seed=123,
        ds_texture_size=4,
        ds_background_noise=0,
        ds_mean=0,
        ds_mean_std=0
    )
    batch = ds.sample(batch_size=4)
    for k, v in batch.items():
        if isinstance(v, int):
            print(k, v)
        else:
            print(k, v.size())
    print('ds.meta', ds.meta)


@pytest.mark.skip()
def test_hypotheses_single_ds_colors():
    ds = TexRelDataset(
        ds_refs=['ds15_color_posneg'],
        ds_filepath_templ='~/data/reftask/{ds_ref}.dat',
        ds_seed=123,
        ds_texture_size=4,
        ds_background_noise=0,
        ds_mean=0,
        ds_mean_std=0
    )
    batch = ds.sample(batch_size=4)
    for k, v in batch.items():
        if isinstance(v, int):
            print(k, v)
        else:
            print(k, v.size())
    print('ds.meta', ds.meta)


@pytest.mark.skip()
def test_hypotheses_fused():
    ds = TexRelDataset(
        ds_refs=['ds19_rels_posneg_nodist', 'ds15_color_posneg'],
        ds_filepath_templ='~/data/reftask/{ds_ref}.dat',
        ds_seed=123,
        ds_texture_size=4,
        ds_background_noise=0,
        ds_mean=0,
        ds_mean_std=0
    )
    batch = ds.sample(batch_size=4)
    for k, v in batch.items():
        if isinstance(v, int):
            print(k, v)
        else:
            print(k, v.size())
    print('hyp', batch['hypotheses_t'].transpose(0, 1))
    print('ds.meta', ds.meta)
