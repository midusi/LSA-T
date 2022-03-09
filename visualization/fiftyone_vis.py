import fiftyone as fo
import argparse, json
from pathlib import Path
from fiftyone import Sample


def get_score(scores):
    m1 = max(scores)
    scores = scores.copy()
    scores.remove(m1)
    return 0 if m1 == 0 else (m1 - max(scores)) / m1

def store_sample(clip_file: Path, dataset):
    print(clip_file)
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

def main():
    parser = argparse.ArgumentParser(description='''Generates DB able to visualize clips on fiftyone and starts fiftyone.''')
    parser.add_argument('--generate', '-g', help='generates the database, only necessary the first time running the script', action='store_true')
    must_gen: bool = parser.parse_args().generate

    if must_gen:
        fo.load_dataset("cn_sordos").delete()
        dataset = fo.Dataset("cn_sordos", persistent=True)
        path = Path("./data/cuts")
        clips = list(path.rglob("*.mp4"))
        for i,c in enumerate(clips,1):
            print(f"{i}/{len(clips)}")
            store_sample(c, dataset)
    else:
        dataset = fo.load_dataset("cn_sordos")

    # View summary info about the dataset
    print(dataset)
    sess = fo.launch_app(dataset=dataset)
    sess.wait()

if __name__ == "__main__":
    main()
