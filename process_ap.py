import json
import os
from typing import TypedDict
import numpy as np


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
        added = False
        for g in grouped:
            if not added and (not g or g[-1]['image_id'] != kd['image_id']):
                g.append(kd)
                added = True
        if not added:
            grouped.append([kd])
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

path = "data/cuts/"
has_ap = lambda c: os.path.isfile(path + c + "/alphapose-results.json")
cuts = [(path+vid+'/'+cut) for vid in os.listdir(path) for cut in os.listdir(path + vid) if has_ap(vid+'/'+cut)]

# Identify signers
for idx, cut in enumerate(cuts):
    if not os.path.isdir(cut):
        continue

    print("{}/{}: {}".format(idx + 1, len(cuts), cut))

    with open(path + "/alphapose-results.json") as ap_file:
        # signers contains a list of lists of keypoints data, one for each signer
        ap = json.load(ap_file)
        signers: list[list[KeypointData]] = group_kds(ap)

    keypoints_for_signers: list[dict[str,list[list]]] = []
    for s in signers:
        keypoints_for_signers.append({
            'x': [[] for _ in range(136)],
            'y': [[] for _ in range(136)],
            'c': [[] for _ in range(136)]
        })
        for keydata in s:
            for idx, keypoint in enumerate(keydata["keypoints"]):
                if idx % 3 == 0:
                    keypoints_for_signers[-1]['x'][int(idx/3)].append(keypoint)
                if idx % 3 == 1:
                    keypoints_for_signers[-1]['y'][int(idx/3)].append(keypoint)
                if idx % 3 == 2:
                    keypoints_for_signers[-1]['c'][int(idx/3)].append(keypoint)

    scores = []
    for idx, each in enumerate(keypoints_for_signers):
        distance_x = np.array(list(map(lambda keys: max(keys) - min(keys), each['x'])))
        distance_y = np.array(list(map(lambda keys: max(keys) - min(keys), each['y'])))
        distance = np.sqrt(distance_x**2 + distance_y**2)
        #confidence = np.array(list(map(max, each['c'])))
        total = np.sum(distance[:94])
        scores.append(total)

    signer = signers[scores.index(max(scores))]

    with open(cut + "_signer.json", "w", encoding="utf-8") as signer_file:
        json.dump({
            "roi": get_box(signer),
            "keypoints": signer,
            "scores": scores
        }, signer_file)
        
    with open(path + "_ap.json", 'w') as ap_file:
        json.dump(ap, ap_file)
        os.remove(path + "/alphapose-results.json")
        os.rmdir(path)
