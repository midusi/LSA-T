import json
import os
from itertools import takewhile
from typing import TypedDict, Iterable, TypeVar
from statistics import mean
import numpy as np

T = TypeVar("T")

class Box(TypedDict):
    x1: float
    y1: float
    width: float
    height: float

class KeypointData(TypedDict):
    image_id: str
    category_id: int
    keypoints: list[float]
    score: float
    box: list[float]
    idx: list[float]

def closest_box(positions: list[Box], b: Box) -> int:
    '''Given a list of signers boxes and a box b returns the index of the closest box from the list to b'''
    min_idx, min_diff = 0, None
    for idx, pos in enumerate(positions):
        diff = abs(b['x1'] - pos['x1']) + abs(b['y1'] - pos['y1'])
        if min_diff is None or min_diff > diff:
            min_idx, min_diff = idx, diff
    return min_idx

def grouped(iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
    "(s, n) -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)]*n)

def format_box(box: list[float]) -> Box:
    return {
        'x1': box[0],
        'y1': box[1],
        'width': box[2],
        'height': box[3]
    }

# Identify signer
for f in os.listdir("res"):

    if f.startswith('.'):
        continue

    with open("res/" + f + "/alphapose-results.json") as res_file:
        res = json.load(res_file)

    # seqs contains a list of lists of keypoints data, one for each signer
    seqs: list[list[KeypointData]] = []

    first_boxes = list(map(lambda kd: format_box(kd["box"]), (takewhile(lambda kd: kd["image_id"] == "0.jpg", res))))
    for b in first_boxes:
        seqs.append([])

    grouped_kds: list[tuple[KeypointData, ...]] = list(grouped(res, len(first_boxes)))
    for kds in grouped_kds:
        for kd in kds:
            seqs[closest_box(first_boxes, format_box(kd["box"]))].append(kd)


    keypoints: list[dict[str,list[list]]] = []
    for s in seqs:
        keypoints.append({
            'x': [[] for _ in range(136)],
            'y': [[] for _ in range(136)],
            'c': [[] for _ in range(136)]
        })
        for keydata in s:
            for idx, keypoint in enumerate(keydata["keypoints"]):
                if idx % 3 == 0:
                    keypoints[-1]['x'][int(idx/3)].append(keypoint)
                if idx % 3 == 1:
                    keypoints[-1]['y'][int(idx/3)].append(keypoint)
                if idx % 3 == 2:
                    keypoints[-1]['c'][int(idx/3)].append(keypoint)

    # print(len(seqs), len(seqs[1]), len(seqs[0][0]["keypoints"]))
    # print(len(keypoints), len(keypoints[0]['x']), len(keypoints[0]['x'][0]))

    max_idx, max_total = 0, None
    for idx, each in enumerate(keypoints):
        distance_x = np.array(list(map(lambda keys: max(keys) - min(keys), each['x'])))
        distance_y = np.array(list(map(lambda keys: max(keys) - min(keys), each['y'])))
        distance = np.sqrt(distance_x**2 + distance_y**2)
        confidence = np.array(list(map(mean, each['c'])))
        total = np.sum(np.multiply(distance, confidence))
        #total = np.sum(distance)
        if max_total is None or max_total < total:
            max_idx = idx
            max_total = total

    signer = seqs[max_idx]

    # Crop video
    x1s = []
    y1s = []
    widths = []
    heights = []

    for keydata in signer:
        x1s.append(keydata['box'][0])
        y1s.append(keydata['box'][1])
        widths.append(keydata['box'][2])
        heights.append(keydata['box'][3])

    x1, y1, = mean(x1s), mean(y1s)
    width, height = mean(widths), mean(heights)

    with open("res/" + f + "/box.txt", 'w') as box_f:
        box_f.write('\n'.join(map(str, [x1,y1,width,height])) + '\n')
