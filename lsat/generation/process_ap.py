import json
import argparse
from pathlib import Path
from math import sqrt

from lsat.typing import Box, KeypointData
from lsat.helpers.get_cut_paths import get_cut_paths
from lsat.helpers.group_kds import group_kds


def format_box(box: list[float]) -> Box:
    'Creates box from list of strings'
    return {
        'x1': box[0],
        'y1': box[1],
        'width': box[2],
        'height': box[3]
    }

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
    'Returns relative position of a keypoint respect to a point'
    center_x = box['x1'] + box['width']/2
    center_y = box['y1'] + box['height']/2
    return (center_x - x, center_y - y)

def main():
    'Infers, in case that there is many people detected by AlphaPose in one clip, which one is the signer. Result is stored in clip_signer.json'
    parser = argparse.ArgumentParser(description='''Infers, in case that there is many people detected by AlphaPose in one clip, which one is the signer.''')
    parser.add_argument('--rerun', '-r', help='runs it over all files, even those already processed', action='store_true')
    must_rerun: bool = parser.parse_args().rerun

    path = Path("data/cuts/")
    cuts = map(get_cut_paths, path.glob('**/*.mp4'))

    cuts = list(filter(lambda c: c['ap'].exists() if must_rerun else c['ap_raw'].exists(), cuts))

    # Identify signers
    for i_cut, cut in enumerate(cuts):
        print(f"{i_cut + 1}/{len(cuts)}: {cut['mp4']}")

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
                for i_keyp, keypoint in enumerate(keydata["keypoints"]):
                    if i_keyp % 3 == 0:
                        keypoints_for_signers[-1]['x'][int(i_keyp/3)].append(keypoint)
                    if i_keyp % 3 == 1:
                        keypoints_for_signers[-1]['y'][int(i_keyp/3)].append(keypoint)
                    if i_keyp % 3 == 2:
                        keypoints_for_signers[-1]['c'][int(i_keyp/3)].append(keypoint)

        scores: list[float] = []
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
            scores.append(distance / ((len(each['c'][0])) / step))

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
