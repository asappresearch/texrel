import os
import sys
from os import path
from typing import Dict, Any

import PIL.Image
import PIL.ImageFont
import PIL.ImageDraw

import torch


class KillFileMonitor(object):
    def __init__(self, kill_file, dead_file):
        self.kill_file = kill_file
        self.dead_file = dead_file
        if self.dead_file is not None and path.exists(self.dead_file):
            os.remove(self.dead_file)

    def check(self):
        if self.kill_file is not None and path.isfile(self.kill_file):
            os.remove(self.kill_file)
            with open(self.dead_file, 'w'):
                pass
            print('dieing, from kill file')
            raise Exception('dieing, from kill file')


def save_image(filepath, image, resize=None, text=None, text_size=24, text_color=0, print_filepath=False):
    """
    image should be torch tensor, float, [0-1], [3][H][W]

    (one-channel ok too)

    resize should be a multiplier (ideally integer, but not obligatorily). upsampling is nearest-neighbor
    """
    C, H, W = image.size()
    if image.dtype == torch.float32:
        image = (image * 255).byte()
        text_color = int(text_color * 255)
    if C == 1:
        image = image.expand(3, H, W)

    font_filepath = None
    if font_filepath is None and path.isdir('/Library/Fonts'):
        font_filepath = '/Library/Fonts/Arial.ttf'
    else:
        font_filepath = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    pil_image = PIL.Image.fromarray(image.transpose(0, 1).transpose(1, 2).detach().numpy())
    if resize is not None and resize != 1:
        size = list(pil_image.size)
        size[0] = int(size[0] * resize)
        size[1] = int(size[1] * resize)
        pil_image = pil_image.resize(size)
    if text is not None and text != '':
        draw = PIL.ImageDraw.Draw(pil_image)
        font = PIL.ImageFont.truetype(font_filepath, text_size)
        draw.text((0, 0), text, (text_color, text_color, text_color), font=font)
    pil_image.save(filepath + '.tmp.png')
    os.rename(filepath + '.tmp.png', filepath)
    if print_filepath:
        print(f'saved image to {filepath}')


def save_image_grid(
        filepath, image_grid, margin_size=0, margin_value=0, text=None, texts=None, text_size=24, text_color=0,
        print_filepath=False,
        resize=None
):
    """
    image should be torch tensor, float, [0-1], [GridH][GridW][3][H][W]

    (one-channel ok too)

    texts are image specific
    """
    gridH, gridW, C, H, W = image_grid.size()
    if image_grid.dtype == torch.float32:
        image_grid = (image_grid * 255).byte()
        margin_value = int(margin_value * 255)
        text_color = int(text_color * 255)
    if C == 1:
        image_grid = image_grid.expand(gridH, gridW, 3, H, W)
    flattened_image = torch.zeros(
        3, gridH * H + margin_size * (gridH - 1), gridW * W + margin_size * (gridW - 1), dtype=torch.uint8)
    if margin_value != 0:
        flattened_image.fill_(margin_value)
    for h in range(gridH):
        for w in range(gridW):
            flattened_image[
                :,
                h * H + h * margin_size:
                    (h + 1) * H + h * margin_size,
                w * W + w * margin_size:
                    (w + 1) * W + w * margin_size
            ] = image_grid[h][w]

    font_filepath = None
    if font_filepath is None and path.isdir('/Library/Fonts'):
        font_filepath = '/Library/Fonts/Arial.ttf'
    else:
        font_filepath = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    pil_image = PIL.Image.fromarray(flattened_image.transpose(0, 1).transpose(1, 2).detach().numpy())
    if resize is not None and resize != 1:
        size = list(pil_image.size)
        size[0] = int(size[0] * resize)
        size[1] = int(size[1] * resize)
        pil_image = pil_image.resize(size)
    if text is not None and text != '':
        draw = PIL.ImageDraw.Draw(pil_image)
        font = PIL.ImageFont.truetype(font_filepath, text_size)
        draw.text((10, 0), text, (text_color, text_color, text_color), font=font)

    if texts is not None:
        draw = PIL.ImageDraw.Draw(pil_image)
        font = PIL.ImageFont.truetype(font_filepath, text_size)
        for h in range(gridH):
            for w in range(gridW):
                draw.text((w * W + 10, h * H + 30), texts[h][w], (text_color, text_color, text_color), font=font)

    pil_image.save(filepath + '.tmp.png')
    os.rename(filepath + '.tmp.png', filepath)
    if print_filepath:
        print(f'saved image to {filepath}')


def filter_dict_by_prefix(dict: Dict[str, Any], prefix: str, truncate_prefix: bool = False) -> Dict[str, Any]:
    if truncate_prefix:
        new_dict = {k.replace(prefix, '', 1): v for k, v in dict.items() if k.startswith(prefix)}
    else:
        new_dict = {k: v for k, v in dict.items() if k.startswith(prefix)}
    return new_dict


def reverse_args(args, neg_key, pos_key):
    assert pos_key not in args.__dict__
    if neg_key in args.__dict__:
        args.__dict__[pos_key] = not args.__dict__[neg_key]
        del args.__dict__[neg_key]


def die():
    import sys
    import inspect
    callerframerecord = inspect.stack()[1]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    print('EXITING, file', info.filename, 'line', info.lineno)
    sys.exit(-1)


def expand(path):
    return path.replace('~', os.environ['HOME'])


def either_not_both(a, b):
    return bool(a) ^ bool(b)


def decode_word_idxes(vocab, word_idxes):
    return ' '.join([vocab[idx] for idx in word_idxes])


class Vocab(object):
    """
    Concept/api, and probably implementation,
    adapted/copied from Index
    https://github.com/jacobandreas/l3/blob/a76d8e99d887bcec6c3010b337e07f262ff83e2c/misc/util.py#L40-L70
    (Apache 2 license)

    (I wrote this from memory, but after looking at the original implementation, and liking
    it a lot :)

    name w2i adapted from some code I saw Lili Yu write
    """
    def __init__(self, words=None):
        if words is None:
            words = []
        self.words = words
        self.w2i = {w: i for i, w in enumerate(self.words)}
        self.size = len(self.w2i)
        self.frozen = False

    def freeze(self, unknown_tok='<unk>'):
        """
        all unknown words after this will go to <unk> (which should already be in the vocab)
        """
        if unknown_tok is not None:
            assert unknown_tok in self.w2i
        self.unknown_tok = unknown_tok
        self.frozen = True

    def _add_word(self, word):
        self.w2i[word] = len(self.w2i)
        self.words.append(word)
        self.size = len(self.w2i)

    def index_of(self, word):
        if word not in self.w2i:
            if self.frozen:
                return self.w2i(self.unknown_tok)
            else:
                self._add_word(word)
        return self.w2i[word]

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return self.words[i]

    def __str__(self):
        return str(self.w2i)


def clean_argv():
    changes = {}
    for i, v in enumerate(sys.argv):
        if isinstance(v, str):
            if v.endswith('\ufeff'):
                print('(debug: cleaned utf-8 bom from args)')
                v = v[:-1]
                changes[i] = v
    for i, v in changes.items():
        sys.argv[i] = v
