from typing import TypedDict


class Box(TypedDict):
    'Standard box data format for ROI'
    x1: float
    y1: float
    width: float
    height: float
