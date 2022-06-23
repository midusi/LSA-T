import csv
from pathlib import Path
from math import ceil, floor
from typing import Callable


def split_train_test(path: Path, filter_sample: Callable[[Path], bool] = lambda x: True) -> tuple[list[Path], list[Path]]:
    '''From root of the database builds train and test sets'''
    video_paths = [video for playlist in path.glob('**/*') for video in playlist.glob('**/*')]
    train_samples: list[Path] = []
    test_samples: list[Path] = []
    for video in video_paths:
        clips = list(
            filter(filter_sample,
            map(lambda vid_path: Path(str(vid_path.resolve())[:-3] + "json"), video.glob('**/*.mp4'))))
        train_samples += clips[:ceil(len(clips) * 0.8)]
        test_samples += clips[-floor(len(clips) * 0.2):]
    return (train_samples, test_samples)

def load_train_test(train_path: Path, test_path: Path) -> tuple[list[Path], list[Path]]:
    '''Loads train and test sets from csv file'''
    train_samples = None
    test_samples = None
    with train_path.open() as train_f:
        train_samples = list(map(Path, list(csv.reader(train_f))[0]))
    with test_path.open() as test_f:
        test_samples = list(map(Path, list(csv.reader(test_f))[0]))
    return (train_samples, test_samples)

def store_samples_to_csv(path: Path, samples: list[Path]):
    '''Stores list of samples to csv file'''
    with path.open('w') as train_file:
        csv.writer(train_file, quoting=csv.QUOTE_ALL).writerow(map(lambda s: s.absolute(), samples))
