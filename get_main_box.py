import json
import os
from typing import TypedDict
from statistics import mean
import numpy as np

class KeypointData(TypedDict):
    image_id: str
    category_id: int
    keypoints: list[float]
    score: float
    box: tuple[float,float,float,float]
    idx: list[float]

# Identify signer

for f in os.listdir("res"):

    if f.startswith('.'):
        continue

    with open("res/" + f + "/alphapose-results.json") as res_file:
        res = json.load(res_file)

    # seqs contains a list of lists of keypoints data, one for each signer
    seqs: list[list[KeypointData]] = [[]]
    for r in res:
        for idx, s in enumerate(seqs):
            if idx == len(seqs) - 1:
                seqs.append([])
            if  not s or r["image_id"] != s[-1]["image_id"]:
                s.append(r)
                break

    seqs = seqs[:-1]

    keypoints: list[dict[list[list]]] = []
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

    #print(len(seqs), len(seqs[0]), len(seqs[0][0]["keypoints"]))
    #print(len(keypoints), len(keypoints[0]['x']), len(keypoints[0]['x'][0]))

    max_idx, max_total = None, None
    for idx, signer in enumerate(keypoints):
        distance_x = np.array(list(map(lambda keys: max(keys) - min(keys), signer['x'])))
        distance_y = np.array(list(map(lambda keys: max(keys) - min(keys), signer['y'])))
        distance = np.sqrt(distance_x**2 + distance_y**2)
        confidence = np.array(list(map(mean, signer['c'])))
        total = np.sum(np.multiply(distance, confidence))
        if not max_total or max_total < total:
            max_idx = idx

    signer = seqs[max_idx]

    # Crop video

    box = {
        'x1': [],
        'y1': [],
        'width': [],
        'height': []
    }
    for keydata in signer:
        box['x1'].append(keydata['box'][0])
        box['y1'].append(keydata['box'][1])
        box['width'].append(keydata['box'][2])
        box['height'].append(keydata['box'][3])

    x1, y1, = min(box['x1']), min(box['y1'])
    width, height = max(box['width']), max(box['height'])

    with open("res/" + f + "/box.txt", 'w') as box_f:
        box_f.write('\n'.join(map(str, [x1,y1,width,height])) + '\n')
    
    #bashCommand = "ffmpeg -i \"res/{}/AlphaPose_{}.mp4\" -filter:v \"crop={}:{}:{}:{}\" out.mp4".format(f,f,x1,y1,width,height)
