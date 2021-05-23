"""
handle printing things at different coordinates in a 'screen' of ascii characters.

Is this the best approach? I dont know :)
"""
from colorama import Fore


def apparent_len(text):
    """
    takes into account colorama Fore colors
    returns the width assuming we removed the Fore color characters
    """
    plain_text = text
    while chr(27) in plain_text:
        pos = plain_text.find(chr(27))
        plain_text = plain_text[:pos] + plain_text[pos + 5:]
    return len(plain_text)


def apparent_substr(text, start_pos, end_pos_exclusive):
    """
    like substr, but takes into account that there may be hidden
    colorama codes underneath
    """
    # pos = 0
    pos = 0
    cut_str = ''
    in_section = False
    apparent_pos = 0
    while pos < len(text):
        in_section = apparent_pos >= start_pos and apparent_pos < end_pos_exclusive
        if in_section:
            if text[pos] == chr(27):
                cut_str += text[pos:pos+5]
                pos += 5
                continue
            else:
                cut_str += text[pos]
                pos += 1
                apparent_pos += 1
                continue
        else:
            if text[pos] == chr(27):
                pos += 5
                continue
            else:
                pos += 1
                apparent_pos += 1
                continue
    cut_str += Fore.RESET
    return cut_str


class AsciiScreen(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.rows = []
        for h in range(height):
            # row
            row = ' ' * width
            # for w in range(width):
            # row.append(' ')
            self.rows.append(row)
        # self.rows[3][4] = '*'
        # self.render()

    def render(self):
        print('')
        print('|' + '-' * self.width + '|')
        for row in self.rows:
            print('|' + ''.join(row) + '|')
        print('-' * self.width)
        print('')

    def print(self, y, x, text):
        text = str(text)
        s = text.split('\n')
        if y >= self.height:
            return
        if x >= self.width:
            return
        for h, r in enumerate(s):
            w = apparent_len(r)
            if x + w >= self.width - 1:
                w = self.width - x
                # r = r[:w]
                r = apparent_substr(r, 0, w)
                w = apparent_len(r)
            if x < 0:
                # r = r[-x:]
                r = apparent_substr(r, -x, 10000)
                w = apparent_len(r)
                x = 0
            old = self.rows[y + h]
            # r = apparent_substr(r, -x, 10000)
            new = apparent_substr(old, 0, x) + r + apparent_substr(old, x + w, 10000)
            # new = old[:x] + r + old[x + w:]
            self.rows[y + h] = new
