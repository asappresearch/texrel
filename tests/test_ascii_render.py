from ulfs import ascii_render
from colorama import init as colorama_init, Fore


def test_screen():
    screen = ascii_render.AsciiScreen(53, 118)
    screen.print(5, 10, 'hello')
    screen.print(0, 0, 'hello')
    screen.print(0, 113, 'hello')
    screen.print(3, 115, 'hello')
    screen.print(10, -3, 'hello')
    screen.print(11, -1, 'hello')
    screen.print(15, 18, 'hello\nworld\nmore\nlines\nhere\n')
    screen.print(18, -3, 'hello\nworld\nmore\nlines\nhere\n')
    screen.print(28, 117, 'hello\nworld\nmore\nlines\nhere\n')
    screen.print(40, 5, 23.57)
    screen.print(2, 2, 'hello')
    screen.print(2, 3, 'hello')
    screen.print(2, 5, 'hello')
    screen.print(3, 2, 'hello\nworld')
    screen.print(3, 3, 'hello\nworld')
    screen.print(3, 5, 'hello\nworld')

    class SomeClass(object):
        def __str__(self):
            return 'SomeClass'

    screen.print(45, 5, SomeClass())

    screen.render()


def test_screen_color():
    colorama_init

    screen = ascii_render.AsciiScreen(53, 118)

    screen.print(2, 2, Fore.BLUE + 'hello' + Fore.RESET)
    screen.print(2, 3, Fore.YELLOW + 'hello' + Fore.RESET)
    screen.print(2, 5, Fore.RED + 'hello')

    screen.render()


def test_get_apparent_width():
    text = 'foo' + Fore.RED + 'bar' + Fore.RESET + ' woo'
    print('text', text)
    apparent_width = ascii_render.apparent_len(text)
    print('apparent_width', apparent_width)
    target_width = len('foo') + len('bar') + len(' woo')
    assert target_width == apparent_width


def test_substr_apparent():
    text = 'foo' + Fore.RED + 'bar' + Fore.RESET + ' woo'
    print('text', text)
    print('apparent len', ascii_render.apparent_len(text))
    for i in range(ascii_render.apparent_len(text)):
        for j in range(i + 1, ascii_render.apparent_len(text) + 1):
            print(i, j, ascii_render.apparent_substr(text, i, j))
