import json
from pathlib import Path
from shutil import copyfile
from typing import TypedDict


class KeypointData(TypedDict):
    image_id: str
    category_id: int
    keypoints: list[float]
    score: float
    box: list[float]
    idx: list[float]

class Box(TypedDict):
    x1: float
    y1: float
    width: float
    height: float

class SignerData(TypedDict):
    scores: list[float]
    roi: Box
    keypoints: list[KeypointData]


def get_cut_files(cut: Path) -> dict[str, Path]:
    return {
        'mp4': cut,
        'json': cut.parent / f"{cut.name[:-4]}.json",
        'signer': cut.parent / f"{cut.name[:-4]}_signer.json",
        'ap': cut.parent / f"{cut.name[:-4]}_ap.json"
    }


def main():
    source = Path('./data/cuts')
    out = Path('./data/cuts_only')
    out.mkdir(exist_ok=True,parents=True)
    cuts = source.glob('**/*.mp4')
    for i, cut in enumerate(cuts):
        print(f"Video {i}")
        outpath = out / cut.parent.name
        outpath.mkdir(exist_ok=True,parents=True)
        clip_files = get_cut_files(cut)
        copyfile(cut, (outpath / cut.name))
        with clip_files['signer'].open() as signerf:
            signer: SignerData = json.load(signerf)
        with open(clip_files['json']) as dataf:
            data = json.load(dataf)
            data['scores'] = signer['scores']
            data['roi'] = signer['roi']
        with (outpath / f"{cut.name[:-4]}.json").resolve().open('w') as out_dataf:
            json.dump(data, out_dataf, indent=4)


if __name__ == "__main__":
    main()
