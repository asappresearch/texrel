from ref_task.models import reftask_utils


def test_length_mask():
    batch_size = 16
    utt_len = 5

    mask = reftask_utils.create_length_mask(utt_len=utt_len, batch_size=batch_size)
    print('mask', mask)
