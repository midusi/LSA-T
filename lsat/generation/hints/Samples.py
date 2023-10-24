from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class PersonSample(TypedDict):
	'In "keypoints" contains a matrix of frames x keypoints and in "boxes" a matrix of frames x boxes'
	keypoints: NDArray[np.float16]
	boxes: NDArray[np.float16]
