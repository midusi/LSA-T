import os

import cv2
import numpy as np
from numpy.typing import NDArray


def load_video(path: str) -> tuple[NDArray[np.uint8], int, int]:
	cap = cv2.VideoCapture(path)
	frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
	video = np.empty((frame_count, *size, 3), np.dtype('uint8'))
	for i in range(frame_count):
		_, video[i] = cap.read()
	cap.release()
	return video, frame_rate, frame_count

def store_video(video: NDArray[np.uint8], frame_rate: int, name: str, dir: str = '.temp'):
	if not os.path.exists(dir):
		os.makedirs(dir)
	fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
	out = cv2.VideoWriter(f'{dir}/{name}', fourcc, frame_rate, (video.shape[2], video.shape[1]))
	for frame in video:
		out.write(cv2.resize(frame, (video.shape[2], video.shape[1])))
	out.release()
	cv2.destroyAllWindows()
