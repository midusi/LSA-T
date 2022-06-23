from math import ceil

from typing import Callable, TypeVar
import torch
from torch import Tensor, stack
from torchvision.transforms.functional import crop, resize

from type_hints import Box, KeypointData
from helpers.get_keypoints_centers import get_keypoints_centers, interpolate

T = TypeVar('T')
def get_frames_reduction_transform(max_frames: int) -> Callable[[list[T]], list[T]]:
    '''Given the desired frame amount, returns a transform that reduces amount of frames of sequence to max_frames'''
    def frames_reduction_transform(clip: list[T]) -> list[T]:
        frames: list[T] = []
        for frame in [c for (i,c) in enumerate(clip) if (i%(ceil(len(clip)/max_frames)) == 0)]:
            frames.append(frame)
        if len(frames) < max_frames:
            for _ in range(max_frames - len(frames)):
                frames.append(frames[-1])
        return frames
    return frames_reduction_transform

def get_roi_selector_transform(height: int, width: int) -> Callable[[Tensor, Box], Tensor]:
    '''Given height and width, returns a frame-level transform that crops a given roi from the frame and resizes it to to the desired values keeping the aspect ratio and padding with zeros if necessary'''
    def roi_selector_transform(img: Tensor, box: Box) -> Tensor:
        img = crop(img, int(box['y1']),int(box['x1']),int(box['height']),int(box['width']))
        pad = torch.zeros(3, height, width, dtype=torch.uint8)
        if (box['height'] - height) > (box['width'] - width):
            new_width = int(box['width']*height/box['height'])
            img = resize(img, [height, new_width])
            pad[:, :, int((width - new_width)/2):-int((width - new_width)/2) - (1 if (width - new_width) % 2 == 1 else 0)] = img
        else:
            new_height = int(box['height']*width/box['width'])
            img = resize(img, [new_height, width])
            pad[:, int((height - new_height)/2):-int((height - new_height)/2) - (1 if (height - new_height) % 2 == 1 else 0), :] = img
        return pad
    return roi_selector_transform

def get_keypoint_format_transform(keypoints_to_use: list[int]) -> Callable[[KeypointData], Tensor]:
    def keypoint_format_transform(keypoint_data: KeypointData) -> Tensor:
        return Tensor([[
            k for j,k in enumerate(keypoint_data['keypoints']) if (j%3) == i and int(j/3) in keypoints_to_use
        ] for i in range(3)])
    return keypoint_format_transform

def keypoints_norm_to_nose(keypoints: Tensor) -> Tensor:
    return (keypoints - Tensor([
        [keypoints[0][0].item()],
        [keypoints[1][0].item()],
        [0]
    ]))

def interpolate_keypoints(keypoints: list[Tensor]) -> list[Tensor]:
    # switch dims to keypoints, frames, (x,y,c)
    keypoints_trans = stack(keypoints).permute(2, 0, 1)
    interpolated_keypoints = Tensor([interpolate(each.tolist(), 0.2, 0.7, (0, 0)) for each in keypoints_trans])
    return [
        frame for frame in interpolated_keypoints.permute(1, 2, 0)
    ]

def keypoint_norm_to_center_transform(keypoints: list[KeypointData]) -> list[KeypointData]:
    centers = get_keypoints_centers(keypoints, threshold=0.2, max_missing_percent=0.3)
    keypoints_norm: list[KeypointData] = []
    for i_frame, keypoint_data in enumerate(keypoints):
        keypoint_norm = keypoint_data
        keypoint_norm['keypoints'] = [
            keypoint - centers[i_frame][0] if (i_key%3)==0 else
            keypoint - centers[i_frame][1] if (i_key%3)==1 else
            keypoint
            for i_key, keypoint in enumerate(keypoint_data['keypoints'])
        ]
        keypoints_norm.append(keypoint_norm)
    return keypoints_norm

def get_text_to_tensor_transform(bos_idx: int, eos_idx: int) -> Callable[[list[int]], Tensor]:
    def text_to_tensor_transform(token_ids: list[int]) -> Tensor:
        return torch.cat((torch.tensor([bos_idx]),
                        torch.tensor(token_ids),
                        torch.tensor([eos_idx])))
    return text_to_tensor_transform
