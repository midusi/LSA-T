import argparse, json
import fiftyone as fo
from pathlib import Path
from fiftyone import Sample

from lsat.helpers.get_score import get_score


def store_sample(clip_file: Path, dataset):
    sample = Sample(clip_file)
    with Path(str(clip_file.resolve())[:-3] + "json").open() as sample_data_file:
        sample_data = json.load(sample_data_file)
        sample["video"] = sample_data["video"]
        sample["start"] = sample_data["start"]
        sample["end"] = sample_data["end"]
        sample["roi"] = sample_data["roi"]
        sample["ground_truth"] = fo.Classification(label = sample_data["label"])
        sample["confidence"] = 0 if len(sample_data["scores"]) == 0 else 1 if len(sample_data["scores"]) == 1 else get_score(list(map(float,sample_data["scores"])))
    dataset.add_sample(sample)

def store_full_sample(clip_file: Path, dataset):
    sample = Sample(clip_file)
    with Path(str(clip_file.resolve())[:-3] + "json").open() as sample_data_file:
        sample_data = json.load(sample_data_file)
        sample["video"] = sample_data["video"]
        sample["start"] = sample_data["start"]
        sample["end"] = sample_data["end"]
        sample["ground_truth"] = fo.Classification(label = sample_data["label"])
    with Path(str(clip_file.resolve())[:-4] + "_signer.json").open() as sample_signer_file:
        signer = json.load(sample_signer_file)
        roi = signer["roi"]
        sample["prediction"] = fo.Detections(
            detections=[
                fo.Detection(
                    label="signer",
                    bounding_box = [roi["x1"]/1920, roi["y1"]/1080, roi["width"]/1920, roi["height"]/1080],
                    confidence = 0 if len(signer["scores"]) == 0 else 1 if len(signer["scores"]) == 1 else get_score(list(map(float,signer["scores"])))
                ),
            ]
        )
        for i, keydata in enumerate(signer["keypoints"],1):
            sample.frames[i]["keypoints"] = fo.Keypoints(keypoints=[
                fo.Keypoint(
                    points=[(keydata["keypoints"][i]/1920, keydata["keypoints"][i+1]/1080)],
                    confidence=keydata["keypoints"][i+2]
                )
            for i in range(93, len(keydata["keypoints"]), 3) if keydata["keypoints"][i+2] > 0.5])
    dataset.add_sample(sample)

def gen_fiftyone_visualization():
    parser = argparse.ArgumentParser(description='''Generates DB able to visualize clips on fiftyone and starts fiftyone.''')
    parser.add_argument('--full', '-f', help='loads the full database (hd videos and uses live keypoints and roi data', action='store_true')
    parser.add_argument('--reload', '-r', help='reloads the database', action='store_true')
    full_db: bool = parser.parse_args().full
    reload_db: bool = parser.parse_args().reload

    db_name = "lsa-t" if full_db else "lsa-t_vis"
    path = Path("./data/cuts") if full_db else Path("./data/cuts_visualization")
    try:
        dataset = fo.load_dataset(db_name)
        if reload_db:
            dataset.delete()
            raise Exception()
    except:
        dataset = fo.Dataset(db_name, persistent=True)
        clips = list(path.rglob("*.mp4"))
        for i,c in enumerate(clips,1):
            print(f"{i}/{len(clips)}")
            if full_db:
                store_full_sample(c, dataset)
            else:
                store_sample(c, dataset)

    # View summary info about the dataset
    print(dataset)
    sess = fo.launch_app(dataset=dataset)
    sess.wait()

if __name__ == "__main__":
    gen_fiftyone_visualization()
