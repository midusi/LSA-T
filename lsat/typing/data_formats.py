from typing import TypedDict

from lsat.typing.Box import Box


class KeypointData(TypedDict):
    '''Data format for each frame of the AlphaPose output'''
    image_id: str
    category_id: int
    keypoints: list[float]
    score: float
    box: list[float]
    idx: list[float]

class SignerData(TypedDict):
    '''Data format for generated signer data file'''
    scores: list[float]
    roi: Box
    keypoints: list[KeypointData]

class CutData(TypedDict):
    '''Data format of the cuts json data file'''
    label: str
    start: float
    end: float
    video: str
    playlist: str
