import json, argparse
from pathlib import Path
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

def get_cut_paths(cut: Path) -> dict[str, Path]:
    'Return paths for al the files (if exists) corresponding to a single clip'
    name = cut.name[:-4]
    return {
        'mp4': cut,
        'json': cut.parent / f"{name}.json",
        'signer': cut.parent / f"{name}_signer.json",
        'ap': cut.parent / f"{name}_ap.json",
        'ap_raw': cut.parent / {name} / "alphapose-results.json",
    }

def format_box(box: list[float]) -> Box:
    'Creates box from list of strings'
    return {
        'x1': box[0],
        'y1': box[1],
        'width': box[2],
        'height': box[3]
    }

def group_kds(kds: list[KeypointData]) -> list[list[KeypointData]]:
    'Groups keypoint data objects that belong to same frame'
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
    'Get box of signer alongside the entire clip'
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

def relative_pos(box: Box, x: float, y: float) -> tuple[float, float]:
    'Returns relative position of a keypoint respect to the signers box center'
    center_x = box['x1'] + box['width']/2
    center_y = box['y1'] + box['height']/2
    return (center_x - x, center_y - y)

def main():
    parser = argparse.ArgumentParser(description='''Infers, in case that there is many people detected by AlphaPose in one clip, which one is the signer.''')
    parser.add_argument('--rerun', '-r', help='runs it over all files, even those already processed', action='store_true')
    must_rerun: bool = parser.parse_args().rerun

    path = Path("../data/cuts/")
    cuts = map(get_cut_paths, path.glob('**/*.mp4'))

    cuts = list(filter(lambda c: c['ap'].exists() if must_rerun else c['ap_raw'].exists(), cuts))

    # Identify signers
    for idx, cut in enumerate(cuts):
        print(f"{idx + 1}/{len(cuts)}: {cut}")

        ap_path = cut['ap'] if must_rerun else cut['ap_raw']
        with ap_path.open() as ap_file:
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
                # for each keypoint there is its x and y position and confidence
                xs = each['x'][i_keyp]
                ys = each['y'][i_keyp]
                cs = each['c'][i_keyp]
                for i_frame in range(0, len(cs), step):
                    # we only consider keypoints with confidence greater than 0.5
                    if i_frame + step < len(cs) and cs[i_frame] > 0.5 and cs[i_frame + step] > 0.5:
                        box1, box2 = signers[i_signer][i_frame]['box'], signers[i_signer][i_frame + step]['box']
                        rel_x1, rel_y1 = relative_pos(format_box(box1), xs[i_frame], ys[i_frame])
                        rel_x2, rel_y2 = relative_pos(format_box(box2), xs[i_frame + step], ys[i_frame + step])
                        distance += sqrt((rel_x1 - rel_x2)**2 + (rel_y1 - rel_y2)**2)
            # movement is normalized respect to the amount of frames
            scores.append(distance / len(cs))

        signer = signers[scores.index(max(scores))]

        with cut['signer'].open(mode='w', encoding='utf-8') as signer_file:
            json.dump({
                "scores": scores,
                "roi": get_box(signer),
                "keypoints": signer
            }, signer_file, indent=4)
        
        if not must_rerun:
            with cut['ap'].open(mode='w') as ap_file:
                json.dump(ap, ap_file)
                cut['ap_raw'].unlink()
                cut['ap_raw'].parent.rmdir()

if __name__ == "__main__":
    main()
