"""
build dataset from a particular family
make dataset metadata available in cross-family way
"""
from texrel.dataset_runtime import TexRelDataset
from shapeworld_data.dataset_runtime_origdata_novgg import Dataset as ShapeworldRuntime


def build_family(ds_family: str, **ds_kwargs):
    if ds_family == 'texrel':
        if ds_kwargs['ds_refs'] is not None:
            return TexRelDataset(
                **{k: ds_kwargs[k] for k in [
                    'ds_filepath_templ', 'ds_seed', 'ds_texture_size', 'ds_background_noise',
                    'ds_mean', 'ds_mean_std',
                    'ds_refs', 'ds_val_refs']}
            )
        return TexRelDataset.from_collection(
            **{k: ds_kwargs[k] for k in [
                'ds_filepath_templ', 'ds_seed', 'ds_texture_size', 'ds_background_noise',
                'ds_mean', 'ds_mean_std',
                'ds_collection', 'ds_tasks', 'ds_distractors', 'ds_val_tasks', 'ds_val_distractors']}
        )
    else:
        return ShapeworldRuntime(data_folder=ds_kwargs['ds_shapeworld_folder'])
