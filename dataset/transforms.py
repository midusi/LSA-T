from math import ceil
from typing import Callable, TypeVar, Optional

import torch
from torch import Tensor, stack
from torchvision.transforms.functional import crop, resize
from torchtext.vocab import Vocab

from type_hints import Box, KeypointData

T = TypeVar('T')
def get_frames_reduction_transform(max_frames: int) -> Callable[[list[T]], list[T]]:
    '''Given the desired frame amount, returns a frames reductor transform'''
    def frames_reduction_transform(clip: list[T]) -> list[T]:
        '''Reduces amount of frames of sequence to max_frames'''
        frames: list[T] = []
        for frame in [c for (i,c) in enumerate(clip) if (i%(ceil(len(clip)/max_frames)) == 0)]:
            frames.append(frame)
        if len(frames) < max_frames:
            for _ in range(max_frames - len(frames)):
                frames.append(frames[-1])
        return frames
    return frames_reduction_transform

def get_roi_selector_transform(height: int, width: int) -> Callable[[Tensor, Box], Tensor]:
    '''Given height and width, returns a frame-level roi selector transform'''
    def roi_selector_transform(img: Tensor, box: Box) -> Tensor:
        '''Frame-level transform that crops a given roi from the frame and resizes it to to the desired values keeping the aspect ratio and padding with zeros if necessary'''
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
    '''Given list of indices of the keypoints to use, returns a KeypointData to tensor transform'''
    def keypoint_format_transform(keypoint_data: KeypointData) -> Tensor:
        '''Using a list of K keypoint indices, transforms a KeypointData item to a tensor with shape (3, K). Each of the K columns has x, y and confidence'''
        return Tensor([[
            k for j,k in enumerate(keypoint_data['keypoints']) if (j%3) == i and int(j/3) in keypoints_to_use
        ] for i in range(3)])
    return keypoint_format_transform

def keypoints_norm_to_nose_transform(keypoints: Tensor) -> Tensor:
    '''Normalizes keypoints (in format given by keypoint_format_transform) to nose keypoint (index 0 using halpe format)'''
    return (keypoints - Tensor([
        [keypoints[0][0].item()],
        [keypoints[1][0].item()],
        [0]
    ]))

def __get_interpolated_point__(i: int, points: list[tuple[float, float, float]], threshold: float, default: tuple[float, float] = (0,0)) -> tuple[float, float]:
    '''Returns for a point, if confidence lower than threshold, the interpolation of the next and previous point with confidence over threshold'''
    next_point = next(((point[0], point[1]) for point in points[(i+1):] if point[2] > threshold), None)
    prev_point = next(((point[0], point[1]) for point in reversed(points[:i]) if point[2] > threshold), None)
    return ((prev_point[0]+next_point[0])/2, (prev_point[1]+next_point[1])/2) if (prev_point is not None and next_point is not None) else (
        next_point if next_point is not None else (
            prev_point if prev_point is not None else default
        )
    )

def __interpolate_each__(keypoints: list[tuple[float, float, float]], threshold: float, max_missing_percent: float, default: Optional[tuple[float, float]] = None) -> Optional[list[tuple[float, float]]]:
    '''For a list of points, replaces those with confidence lower than threshold with the interpolation of the next and previous point with confidence over threshold'''
    # keypoints contains [x,y,z] for each frame 
    missing = sum(1 for point in keypoints if point[2] < threshold)
    if missing / len(keypoints) <= max_missing_percent:
        return [
            (each[0], each[1]) if each[2] > threshold else __get_interpolated_point__(i, keypoints, threshold) for i, each in enumerate(keypoints)
        ]
    return None if not default else [default for _ in keypoints]

def interpolate_keypoints_transform(keypoints: list[Tensor]) -> list[Tensor]:
    '''For a list of keypoint frames (each in format given by keypoint_format_transform), applies __interpolate_each__ to each frame'''
    # switch dims to keypoints, frames, (x,y,c)
    keypoints_trans = stack(keypoints).permute(2, 0, 1)
    interpolated_keypoints = Tensor([__interpolate_each__(each.tolist(), 0.2, 0.7, (0, 0)) for each in keypoints_trans])
    return [
        frame for frame in interpolated_keypoints.permute(1, 2, 0)
    ]

def get_label_to_tensor_transform(bos_idx: int, eos_idx: int, tokenizer: Callable[[str], list[str]], vocab: Vocab) -> Callable[[str], Tensor]:
    '''Returns a label to tensor transform using given tokenizer and vocab'''
    def label_to_tensor_transform(label: str) -> Tensor:
        '''Tokenizes label and transforms it to tensor of the token indices'''
        return torch.cat((torch.tensor([bos_idx]),
                        torch.tensor(vocab.lookup_indices(tokenizer(label))),
                        torch.tensor([eos_idx])))
    return label_to_tensor_transform
