import json
import os
from typing import TypedDict, TypeVar
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

def group_kds(kds: list[KeypointData]) -> list[list[KeypointData]]:
    '''Groups keypoint data objects that belong to same frame'''
    grouped: list[list[KeypointData]] = [[]]
    for kd in kds:
        for g in grouped:
            if not g or g[-1]['image_id'] != kd['image_id']:
                g.append(kd)
                break
    return grouped

def get_box(signer: list[KeypointData]) -> Box:
    box: Box = {
        'x1': signer[0]['box'][0],
        'y1': signer[0]['box'][1],
        'width': signer[0]['box'][2],
        'height': signer[0]['box'][3]
    }

    for keydata in signer:
        box['x1'] = min(box['x1'], keydata['box'][0])
        box['y1'] = min(box['y1'], keydata['box'][1])
        box['width'] = max(box['width'], keydata['box'][2])
        box['height'] = max(box['height'], keydata['box'][3])

    return box

# Identify signers
#for cut in os.listdir("data/cuts"):
#    for f in os.listdir("data/cuts/" + cut):
for f in [d for d in os.listdir("test") if os.path.isdir("test/" + d)]:
    #path = "data/cuts/" + cut + '/' + f
    path = "test/" + f
    print(path)

    with open(path + "/alphapose-results.json") as ap_file:
        # seqs contains a list of lists of keypoints data, one for each signer
        seqs: list[list[KeypointData]] = group_kds(json.load(ap_file))

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
        if max_total is None or max_total < total:
            max_idx = idx
            max_total = total

    signer = seqs[max_idx]

    with open(path + ".json", "r+", encoding="utf-8") as res_file:
        res = json.load(res_file)
        res["roi"] = get_box(signer)
        res["keypoints"] = signer
        res_file.seek(0)
        json.dump(res, res_file)
        res_file.truncate()
        
    os.remove(path + "/alphapose-results.json")
    os.rmdir(path)