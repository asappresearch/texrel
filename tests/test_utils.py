import torch

from ulfs import utils


def test_vocab():
    vocab = utils.Vocab()
    sentence = 'the quick brown fox jumps over the lazy dog'
    tokens = list(sentence)
    N = len(tokens)
    idxes = torch.zeros(N, dtype=torch.int64)
    for n, token in enumerate(tokens):
        idxes[n] = vocab.index_of(token)
    print('idxes', idxes)
    print('vocab.words', vocab.words)
    print('vocab.w2i', vocab.w2i)
    print('len(vocab)', len(vocab))
    print('vocab[0]', vocab[0])
    print('vocab[1]', vocab[1])


def test_save_image_text():
    size = 300
    image = torch.randn(3, size, size) * 0.1 + 0.2
    utils.save_image(filepath='/tmp/out.png', image=image, text='hello world!', text_color=1)


def test_save_image_grid_with_texts():
    size = 224
    gridW = 4
    gridH = 4
    image = torch.randn(gridH, gridW, 3, size, size) * 0.1 + 0.2
    utils.save_image_grid(
        image_grid=image, filepath='/tmp/out.png', margin_size=1, margin_value=0.5, texts=[
            ['acc=8.34 e=1234'] * gridW] * gridH)
