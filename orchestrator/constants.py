from enum import Enum


class Colors(Enum):
    # Regular colors
    black = "\x1b[30m"
    red = "\x1b[31m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    blue = "\x1b[34m"
    magenta = "\x1b[35m"
    cyan = "\x1b[36m"
    white = "\x1b[37m"
    # Bright colors
    bright_black = "\x1b[90m"
    bright_red = "\x1b[91m"
    bright_green = "\x1b[92m"
    bright_yellow = "\x1b[93m"
    bright_blue = "\x1b[94m"
    bright_magenta = "\x1b[95m"
    bright_cyan = "\x1b[96m"
    bright_white = "\x1b[97m"
    # Reset
    reset = "\x1b[0m"
