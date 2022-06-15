import json
from pathlib import Path
from shutil import copyfile

from helpers.get_cut_paths import get_cut_paths
from type_hints import SignerData


def gen_cuts_only_db():
    'Generates a database in data/cuts_only that contains only the videos and metadata (excluding keypoint info)'
    source = Path('./data/cuts')
    out = Path('./data/cuts_only')
    out.mkdir(exist_ok=True,parents=True)
    cuts = source.glob('**/*.mp4')
    for i, cut in enumerate(cuts):
        print(f"Video {i}")
        outpath = out / cut.parent.name
        outpath.mkdir(exist_ok=True,parents=True)
        clip_files = get_cut_paths(cut)
        copyfile(cut, (outpath / cut.name))
        with clip_files['signer'].open() as signerf:
            signer: SignerData = json.load(signerf)
        with (clip_files['json']).open() as dataf:
            data = json.load(dataf)
            data['scores'] = signer['scores']
            data['roi'] = signer['roi']
        with (outpath / f"{cut.name[:-4]}.json").resolve().open('w') as out_dataf:
            json.dump(data, out_dataf, indent=4)


if __name__ == "__main__":
    gen_cuts_only_db()
