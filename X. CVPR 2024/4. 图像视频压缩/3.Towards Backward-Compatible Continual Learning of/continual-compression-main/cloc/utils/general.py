import os
import re
import json
import random
import logging
from pathlib import Path
from tempfile import gettempdir
from collections import OrderedDict


__all__ = [
    'ANSI', 'my_stream_handler',
    'query_yes_no', 'increment_dir', 'parse_config_str', 'random_string',
    'get_temp_file_path', 'read_file', 'json_load', 'json_dump', 'print_to_file',
    'FunctionRegistry', 'SimpleConfig', 'SimpleTable', 'dict_to_table', 'print_dict_as_table'
]


def docstring_example():
    """ A dummy function to show the docstring format that can be parsed by Pylance. \\
    Hyperlink https://github.com \\
    Hyperlink with text [GitHub](https://github.com) \\
    Code `utils.general`

    Args:
        xxxx (type): xxxx xxxx xxxx xxxx.
        xxxx (type, optional): xxxx xxxx xxxx xxxx. Defaults to 'xxxx'.

    ### Bullets:
        - xxxx
        - xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx \
        xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx \
        xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx

    ### Code examples:
        >>> torch.load('tensors.pt')
        # Load all tensors onto the CPU
        >>> with open('tensor.pt', 'rb') as f:
        ...     buffer = io.BytesIO(f.read())

    ### Code examples::

        # comment
        model = nn.Sequential(
            nn.Conv2d(1,20,5),
            nn.ReLU(),
            nn.Conv2d(20,64,5),
            nn.ReLU()
        )
    """
    return 0


class ANSI():
    """ ANSI escape codes with colorizing functions

    Reference:
    - https://en.wikipedia.org/wiki/ANSI_escape_code
    - https://www.lihaoyi.com/post/BuildyourownCommandLinewithANSIescapecodes.html
    """
    # basic colors
    black   = '\u001b[30m'
    red     = r = '\u001b[31m'
    green   = g = '\u001b[32m'
    yellow  = y = '\u001b[33m'
    blue    = b = '\u001b[34m'
    magenta = m = '\u001b[35m'
    cyan    = c = '\u001b[36m'
    white   = w = '\u001b[37m'
    # bright colors
    bright_black   = '\u001b[90m'
    bright_red     = br_r = '\u001b[91m'
    bright_green   = br_g = '\u001b[92m'
    bright_yellow  = br_y = '\u001b[93m'
    bright_blue    = br_b = '\u001b[94m'
    bright_magenta = br_m = '\u001b[95m'
    bright_cyan    = br_c = '\u001b[96m'
    bright_white   = br_w = '\u001b[97m'
    # background colors
    background_black   = '\u001b[40m'
    background_red     = bg_r = '\u001b[41m'
    background_green   = bg_g = '\u001b[42m'
    background_yellow  = bg_y = '\u001b[43m'
    background_blue    = bg_b = '\u001b[44m'
    background_magenta = bg_m = '\u001b[45m'
    background_cyan    = bg_c = '\u001b[46m'
    background_white   = bg_w = '\u001b[47m'
    # misc
    end       = '\u001b[0m'
    bold      = '\u001b[1m'
    underline = udl = '\u001b[4m'
    all_colors_short = [
        'black',               'r',    'g',    'y',    'b',    'm',    'c',    'w',
        'bright_black',     'br_r', 'br_g', 'br_y', 'br_b', 'br_m', 'br_c', 'br_w',
        'background_black', 'bg_r', 'bg_g', 'bg_y', 'bg_b', 'bg_m', 'bg_c', 'bg_w',
    ]
    all_colors_long = [
        'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white',
        'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
        'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white',
        'background_black', 'background_red', 'background_green', 'background_yellow',
        'background_blue', 'background_magenta', 'background_cyan', 'background_white'
    ]

    @classmethod
    def colorstr_example(cls):
        for c in cls.all_colors_long:
            line = ''.join([
                cls.colorstr(f'{c}',                c=c, b=False, ul=False), ', ',
                cls.colorstr(f'{c} bold',           c=c, b=True,  ul=False), ', ',
                cls.colorstr(f'{c} underline',      c=c, b=False, ul=True), ', ',
                cls.colorstr(f'{c} bold underline', c=c, b=True,  ul=True),
            ])
            print(line)

    @classmethod
    def colorstr(cls, msg: str, c='b', b=False, ul=False):
        """ Colorize a string. 

        Args:
            msg (str): string
            c (str): color. Examples: 'red', 'r', 'br_r', ...
            b (bool): bold
            ul (bool): underline
        """
        msg = str(msg)
        if c is not None:
            # msg = eval(f'cls.{c}') + msg
            msg = getattr(cls, c) + msg
        if b:
            msg = cls.bold + msg
        if ul:
            msg = cls.underline + msg
        msg = msg + cls.end
        return msg

    @classmethod
    def printc(cls, *strings, c='blue', b=False, ul=False, **kwargs):
        """ Print with color and style

        Args:
            msg (str): string
            c (str): color. Examples: 'red', 'r', 'br_r', ...
            b (bool): bold
            ul (bool): underline
        """
        strings = [cls.colorstr(s, c, b, ul) for s in strings]
        print(*strings, **kwargs)

    @classmethod
    def errorstr(cls, msg: str):
        msg = cls.bright_red + str(msg) + cls.end
        return msg

    @classmethod
    def warningstr(cls, msg: str):
        msg = cls.yellow + str(msg) + cls.end
        return msg

    @classmethod
    def infostr(cls, msg: str):
        msg = cls.bright_blue + str(msg) + cls.end
        return msg

    @classmethod
    def successstr(cls, msg: str):
        msg = cls.bright_green + str(msg) + cls.end
        return msg
    sccstr = successstr

    @classmethod
    def titlestr(cls, msg: str):
        msg = cls.bold + str(msg) + cls.end
        return msg

    @classmethod
    def headerstr(cls, msg: str):
        msg = cls.underline + str(msg) + cls.end
        return msg

    @classmethod
    def highlightstr(cls, msg: str):
        msg = cls.cyan + str(msg) + cls.end
        return msg
    hlstr = highlightstr

    @classmethod
    def underlinestr(cls, msg: str):
        msg = cls.underline + str(msg) + cls.end
        return msg
    udlstr = underlinestr


def colorstr_example():
    ANSI.colorstr_example()


class LevelFormatter(logging.Formatter):
    _level_formats = {
        logging.WARNING: ANSI.warningstr('[%(asctime)s] %(message)s'),
        logging.ERROR:   ANSI.errorstr('[%(asctime)s] %(message)s'),
    }

    def format(self, record):
        # adapted from https://stackoverflow.com/q/14844970
        # Save the default format configured by the user
        format_default = self._style._fmt
        # Replace the original format with one customized by logging level
        self._style._fmt = self._level_formats.get(record.levelno, format_default)
        # Call the original format method
        result = super().format(record)
        # Restore the original format configured by the user
        self._style._fmt = format_default
        return result

def my_stream_handler():
    handler = logging.StreamHandler()
    formatter = LevelFormatter(fmt='[%(asctime)s] %(message)s', datefmt='%Y-%b-%d %H:%M:%S')
    handler.setFormatter(formatter)
    return handler


def query_yes_no(question):
    """ Ask a yes/no question via input() and return their answer. \\
    The return value is True for 'y' or 'yes', and False for 'n' or 'no'.

    Args:
        question (str): a string that is presented to the user.
    """
    valid = {"yes": True, "y": True, "no": False, "n": False}

    while True:
        print(question + " [y/n]: ", end='')
        choice = input().lower()
        if choice in valid:
            return valid[choice]
        else:
            print("Please respond with yes/no or y/n.")


def increment_dir(dir_root='runs/', name='exp'):
    """ Increament directory name. E.g., exp_1, exp_2, exp_3, ...

    Args:
        dir_root (str, optional): root directory. Defaults to 'runs/'.
        name (str, optional): dir prefix. Defaults to 'exp'.
    """
    assert isinstance(dir_root, (str, Path))
    dir_root = Path(dir_root)
    # if not dir_root.is_dir():
    #     print(f'{dir_root} does not exist. Creating it...')
    #     dir_root.mkdir(parents=True)
    # dnames = [s for s in os.listdir(dir_root) if s.startswith(name)]
    # if len(dnames) > 0:
    #     dnames = [s[len(name):] for s in dnames]
    #     ids = [int(re.search(r'\d+', s).group()) for s in dnames] # left to right search
    #     n = max(ids) + 1
    # else:
    #     n = 0
    n = 0
    while (dir_root / f'{name}_{n}').is_dir():
        n += 1
    name = f'{name}_{n}'
    return name


def random_string(length: int):
    """ Generate a random string of given length.

    Args:
        length (int): length of the string.
    """
    dictionary = 'abcdefghijklmnopqrstuvwxyz0123456789'
    return ''.join(random.choices(dictionary, k=length))


def get_temp_file_path(suffix='.tmp'):
    """ Get a temporary file path.
    """
    tmp_path = Path(gettempdir()) / (random_string(16) + suffix)
    if tmp_path.is_file():
        print(f'{tmp_path} already exists!! Generating another one...')
        tmp_path = Path(gettempdir()) / (random_string(16) + suffix)
    return tmp_path


def warning(msg: str):
    """ Strong warning message

    Args:
        msg (str): warning message
    """
    print('=======================================================================')
    print('Warning:', msg)
    print('=======================================================================')

def warning2(msg: str):
    """ Weak warning message

    Args:
        msg (str): warning message
    """
    msg = ANSI.warningstr(msg)
    print(msg)


def read_file(fpath):
    with open(fpath, mode='r') as f:
        s = f.read()
    return s

def json_load(fpath):
    with open(fpath, mode='r') as f:
        d = json.load(fp=f)
    return d

def json_dump(obj, fpath, indent=2):
    with open(fpath, mode='w') as f:
        json.dump(obj, fp=f, indent=indent)

def print_to_file(msg, fpath, mode='a'):
    with open(fpath, mode=mode) as f:
        print(msg, file=f)


class FunctionRegistry(dict):
    def register(self, func):
        self[func.__name__] = func
        return func


class SimpleConfig(dict):
    """ A simple config class
    """
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


def parse_config_str(config_str: str):
    """

    Args:
        config_str (str): [description]

    ### Examples:
        >>> input_1: 'rand-re0.25'
        >>> output_1: {'rand': True, 're': 0.25}

        >>> input_2: 'baseline'
        >>> output_2: {'baseline': True}
    """
    configs = dict()
    for kv_pair in config_str.split('-'):
        result = re.split(r'(\d.*)', kv_pair)
        if len(result) == 1:
            k = result[0]
            configs[k] = True
        else:
            assert len(result) == 3 and result[2] == ''
            k, v, _ = re.split(r'(\d.*)', kv_pair)
            configs[k] = float(v)
    return configs


def zigzag(n):
    """ Return indices for zig-zag expanding a n x n matrix
    """
    indices = []
    for i in range(2*n-1):
        if i < n:
            for j in range(0,i+1):
                if i % 2 == 0: # i = 0, 2, 4, 6, ...
                    indices.append((i-j,j))
                else: # i = 1, 3, 5, ...
                    indices.append((j,i-j))
        else:
            for j in range(i+1-n, n):
                if i % 2 == 0: # i = 0, 2, 4, 6, ...
                    indices.append((i-j,j))
                else: # i = 1, 3, 5, ...
                    indices.append((j,i-j))
    return indices


def obj_to_str(obj, floatfmt='{:.4g}'):
    """ Convert an object to string

    Args:
        obj (object): object to be converted
        floatfmt (str, optional): float format. Defaults to '{:.4g}'.

    Returns:
        str: string representation of the object
    """
    if isinstance(obj, float) or hasattr(obj, 'float'): # TODO: handle torch.Tensor
        return floatfmt.format(float(obj))
    elif isinstance(obj, (list, tuple)):
        msg = ', '.join([obj_to_str(item, '{:.3g}') for item in obj])
        return f'[{msg}]' if isinstance(obj, list) else f'({msg})'
    else:
        return str(obj)


class SimpleTable(OrderedDict):
    """ A simple table class for printing
    """
    def __init__(self, data: dict={}, floatfmt='{:.4g}'):
        """ Initialize the table

        Args:
            data (dict): initial data. Defaults to an empty dict.
            floatfmt (str): float formatting string. Defaults to '{:.4g}'.
        """
        super().__init__()
        self.update(data)
        self._str_lengths = {k: 6 for k in self.keys()}
        self.floatfmt = floatfmt

    def _update_str_length(self, key, length: int):
        new = max(length, self._str_lengths.get(key, 0))
        self._str_lengths[key] = new
        return new

    def get_header_and_body(self, border=False):
        """ Update the string lengths, and return header and body

        Returns:
            (str, str): (table header, table body)
        """
        head, body = [], []
        for k,v in self.items():
            # convert object to string
            key, val = obj_to_str(k), obj_to_str(v)
            # get str length
            str_len = self._update_str_length(k, max(len(key), len(val)))
            # make header and body string
            head.append(f'{key:^{str_len+2}}')
            body.append(f'{val:^{str_len+2}}')
        head = '|' + '|'.join(head) + '|'
        body = '|' + '|'.join(body) + '|'
        if border:
            head = ANSI.headerstr(head)
        return head, body


def dict_to_table(dictionary: dict):
    """ Convert a dictionary to a table header and body

    Args:
        dictionary (dict): dictionary, usually [str -> float]
        floatfmt (str, optional): float formatting. Defaults to '.4g'.

    Returns:
        (str, str): (table header, table body)
    """
    table = SimpleTable(dictionary)
    return table.get_header_and_body()


def print_dict_as_table(dictionary: dict):
    header, body = dict_to_table(dictionary)
    print(header)
    print(body)


def main():
    from tqdm import tqdm

    stats = {
        'GPU-mem': '3.3G', 'plain_loss': 4.487293014923732,
        'plain_kl': 0.7481001019477844, 'plain_mse': 3.739192873239517,
        'plain_bppix': 3.2378409215057915, 'plain_psnr': 12.756206328649393,
        'plain_ms-ssim': 0.4347556612143914, 'ema_loss': 4.487293014923732,
        'ema_kl': 0.7481001019477844, 'ema_mse': 3.739192873239517,
        'ema_bppix': 3.2378409215057915, 'ema_psnr': 12.756206328649393,
        'ema_ms-ssim': 0.4347556612143914
    }

    print_dict_as_table(stats)

    # pbar = tqdm(range(10000))
    # for _ in pbar:
    #     stats = {k: random.random() for k in stats.keys()}
    #     # header, body = dict_to_table(stats)
    #     header, body = print_dict_as_table(stats)
    #     pbar.set_description(body)


if __name__ == '__main__':
    main()
