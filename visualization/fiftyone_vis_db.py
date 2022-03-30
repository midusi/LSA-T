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

def main():
    parser = argparse.ArgumentParser(description='''Generates DB able to visualize clips on fiftyone and starts fiftyone.''')
    parser.add_argument('--generate', '-g', help='generates the database, only necessary the first time running the script', action='store_true')
    must_gen: bool = parser.parse_args().generate

    if must_gen:
        fo.load_dataset("cn_sordos_vis").delete()
        dataset = fo.Dataset("cn_sordos_vis", persistent=True)
        path = Path("./data/vis_db")
        clips = list(path.rglob("*.mp4"))
        for i,c in enumerate(clips,1):
            print(f"{i}/{len(clips)}")
            store_sample(c, dataset)
    else:
        dataset = fo.load_dataset("cn_sordos_vis")

    # View summary info about the dataset
    print(dataset)
    sess = fo.launch_app(dataset=dataset)
    sess.wait()

if __name__ == "__main__":
    main()
