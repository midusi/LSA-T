import json, argparse, os
from math import sqrt
from typing import TypedDict


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

def relative_pos(box: list[float], x: float, y: float) -> tuple[float, float]:
    center_x = box[0] + box[2]/2
    center_y = box[1] + box[3]/2
    return (abs(center_x - x), abs(center_y - y))

def main():
    parser = argparse.ArgumentParser(description='''Infers, in case that there is many people detected by AlphaPose in one clip, which one is the signer.''')
    parser.add_argument('--rerun', '-r', help='runs it over all files, even those already processed', action='store_true')
    must_rerun: bool = parser.parse_args().rerun

    path = "data/cuts/"
    cuts = [(path+vid+'/'+cut[:-4]) for vid in os.listdir(path) for cut in os.listdir(path + vid) if cut.endswith(".mp4")]

    cuts = list(filter(lambda c: os.path.isfile(c + ("_ap.json" if must_rerun else "/alphapose-results.json")), cuts))

    # Identify signers
    for idx, cut in enumerate(cuts):
        print("{}/{}: {}".format(idx + 1, len(cuts), cut))

        ap_path = cut + ("_ap.json" if must_rerun else "/alphapose-results.json")
        with open(ap_path) as ap_file:
            # signers contains a list of lists of keypoints data, one for each signer
            ap = json.load(ap_file)
            signers: list[list[KeypointData]] = group_kds(ap)

        keypoints_for_signers: list[dict[str,list[list[float]]]] = []
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
        # movement is calculated from frame i to frame i+step
        step = 5
        for i_signer, each in enumerate(keypoints_for_signers):
            distance = 0
            for i_keyp in range(94, len(each['c'])):
                xs = each['x'][i_keyp]
                ys = each['y'][i_keyp]
                cs = each['c'][i_keyp]
                for i_frame in range(0, len(cs), step):
                    if i_frame + step < len(cs) and cs[i_frame] > 0.5 and cs[i_frame + step] > 0.5:
                        box1, box2 = signers[i_signer][i_frame]['box'], signers[i_signer][i_frame + step]['box']
                        rel_x1, rel_y1 = relative_pos(box1, xs[i_frame], ys[i_frame])
                        rel_x2, rel_y2 = relative_pos(box2, xs[i_frame + step], ys[i_frame + step])
                        distance += sqrt((rel_x1 - rel_x2)**2 + (rel_y1 - rel_y2)**2)
            scores.append(distance)

        signer = signers[scores.index(max(scores))]

        with open(cut + "_signer.json", "w", encoding="utf-8") as signer_file:
            json.dump({
                "scores": scores,
                "roi": get_box(signer),
                "keypoints": signer
            }, signer_file, indent=4)
        
        if not must_rerun:
            with open(cut + "_ap.json", 'w') as ap_file:
                json.dump(ap, ap_file)
                os.remove(cut + "/alphapose-results.json")
                os.rmdir(cut)

if __name__ == "__main__":
    main()
