"""
Used for naming logfiles, model files, ...
"""

import datetime
import os
import inspect


def file():
    previous_frame = inspect.currentframe().f_back
    (filename, line_number,
        function_name, lines, index) = inspect.getframeinfo(previous_frame)
    return filename.split('/')[-1].split('.')[0]


def hostname():
    return os.uname().nodename


def date_string():
    return datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d')


def time_string():
    return datetime.datetime.strftime(datetime.datetime.now(), '%H%M%S')
