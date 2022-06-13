from enum import Enum

# TODO: change file name


class Camera(Enum):
    LEFT = "left"
    RIGHT = "right"
    FRONT_WIDE = "front_wide"


class TurnSignal(Enum):
    LEFT = 2
    STRAIGHT = 1
    RIGHT = 0
