from math import sqrt

import h5py
import pandas as pd
import numpy as np
from numpy.typing import NDArray

from hints.Box import Box, format_box


def relative_pos(box: Box, x: float, y: float) -> tuple[float, float]:
    'Returns relative position of a keypoint respect to a point'
    center_x = (box['x1'] + box['x2'])/2
    center_y = (box['y1'] + box['y2'])/2
    return (center_x - x, center_y - y)

KEYPOINTS_PATH = "../data/keypoints.h5"
KEYPOINTS_SIZE = np.float16
STEP = 5

labels = pd.read_csv("../data/labels.csv")
# create empty columns signers_amount, infered_signer and infered_signer_confidence
labels['signers_amount'] = 0
labels['infered_signer'] = ''
labels['infered_signer_confidence'] = .0
labels['movement_per_signer'] = "[]"

with h5py.File(KEYPOINTS_PATH, 'r') as hdf5_file:
	for i, clip in enumerate(labels['id']):
		print(f"{i+1}/{len(labels)}: {clip}")
		clip = f"{clip}.mp4"
		if len(hdf5_file[clip]) == 1:
			labels.loc[labels['id'] == clip, 'signers_amount'] = 1
			labels.loc[labels['id'] == clip, 'infered_signer'] = 0
			labels.loc[labels['id'] == clip, 'infered_signer_confidence'] = 1
		else:
			movement_per_signer = {}
			for signer in hdf5_file[clip]:
				keypoints: NDArray[KEYPOINTS_SIZE] = hdf5_file[clip][signer]['keypoints'][:]
				boxes: NDArray[KEYPOINTS_SIZE] = hdf5_file[clip][signer]['boxes'][:]
				movement = 0
				for i_frame in range(0, len(keypoints), STEP):
					if i_frame + STEP < len(keypoints) and i_frame + STEP < len(boxes):
						# only consider body and hands keypoints
						for i_keypoint in list(range(0, 33)) + list(range(33+468, int(len(keypoints[i_frame])/4))):
							rel_x1, rel_y1 = relative_pos(format_box(boxes[i_frame]), keypoints[i_frame][i_keypoint], keypoints[i_frame][i_keypoint+1])
							rel_x2, rel_y2 = relative_pos(format_box(boxes[i_frame+STEP]), keypoints[i_frame+STEP][i_keypoint], keypoints[i_frame+STEP][i_keypoint+1])
							if not np.isnan(rel_x1) and not np.isnan(rel_x2) and not np.isnan(rel_y1) and not np.isnan(rel_y2):
								movement += sqrt((rel_x1 - rel_x2)**2 + (rel_y1 - rel_y2)**2)
				movement_per_signer[signer] = (movement / ((len(keypoints)) / STEP))
			clip = clip[:-4]
			if len(movement_per_signer) != 0:
				labels.loc[labels['id'] == clip, 'signers_amount'] = len(movement_per_signer)
				labels.loc[labels['id'] == clip, 'infered_signer'] = list(movement_per_signer.items())[list(movement_per_signer.values()).index(max(movement_per_signer.values()))][0]
				labels.loc[labels['id'] == clip, 'infered_signer_confidence'] = (max(movement_per_signer.values()) / sum(movement_per_signer.values())) if sum(movement_per_signer.values()) != 0 else 0
				labels.loc[labels['id'] == clip, 'movement_per_signer'] = str(list(movement_per_signer.values()))
	labels.to_csv("../data/labels_with_signer_data.csv", index=False)