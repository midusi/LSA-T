import json
from pathlib import Path
from urllib.request import urlretrieve
from typing import Callable, Optional, Generator, Literal, Iterable, Iterator

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import VideoReader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import py7zr

from lsat.typing import CutData, SignerData, KeypointData
from lsat.typing import (
    Sample,
    CLIP_HINT,
    KEYPOINTS_HINT,
    LABEL_HINT
)
from lsat.helpers.sample_filters import sample_above_confidence_threshold
from lsat.helpers.get_cut_paths import get_cut_paths
from lsat.helpers.train_test import split_train_test, load_train_test, store_samples_to_csv
from lsat.helpers.ProgressBar import ProgressBar


def _yield_tokens(samples: Iterable[Path], tokenizer: Callable[[str], list[str]]) -> Generator[list[str], None, None]:
    for sample in samples:
        with sample.open() as data_file:
            data: CutData = json.load(data_file)
            yield tokenizer(data['label'])

def _load_clip_as_tensors(paths: dict[str, Path]) -> Iterable[Tensor]:
    return (map(lambda frame: frame['data'], VideoReader(str(paths['mp4']), "video")))


class PyTorchDataset(Dataset):

    def __init__(self,
            root: str,
            mode: Literal["train", "test"],
            load_clips: bool = True,
            load_keypoints: bool = True,
            words_min_freq: int = 1,
            signer_confidence_threshold: float = .5,
            clip_transform: Optional[Callable[[Iterable[Tensor]], CLIP_HINT]] = None,
            keypoints_transform: Optional[Callable[[Iterable[KeypointData]], KEYPOINTS_HINT]] = None,
            label_transform: Optional[Callable[[str], LABEL_HINT]] = None
        ) -> None:
        self.root = Path(root)
        self.mode = mode
        self.load_clips = load_clips
        self.load_keypoints = load_keypoints
        self.words_min_freq = words_min_freq
        self.signer_confidence_threshold = signer_confidence_threshold
        self.clip_transform = clip_transform
        self.keypoints_transform = keypoints_transform
        self.label_transform = label_transform

        if not self.root.exists() or not any(self.root.iterdir()):
            self.root.mkdir(exist_ok=True, parents=True)
            pb = ProgressBar()
            print("Downloading LSA-T")
            urlretrieve("http://c1781468.ferozo.com/data/lsa-t.7z", self.root / "lsat.7z", pb)
            print("Extracting files, this may take a while")
            with py7zr.SevenZipFile(self.root / 'lsat.7z', mode='r') as z:
                z.extractall(self.root)
            
        splits_path = self.root.parent / "splits" / f"confidence_threshold_{str(signer_confidence_threshold).replace('.','')}"
        splits_path.mkdir(exist_ok=True, parents=True)
        train_path = splits_path / "train.csv"
        test_path = splits_path / "test.csv"
        sample_paths = map(lambda p: Path(str(p.resolve())[:-3] + "json"), self.root.glob('**/*.mp4'))
        
        self.tokenizer: Callable[[str], list[str]] = get_tokenizer('spacy', language='es_core_news_lg')
        
        if train_path.exists() and test_path.exists():
            print("Loading existing train and test splits")
            self.train_samples, self.test_samples = load_train_test(train_path, test_path)
        else:
            print("Generating train and test splits")
            self.train_samples, self.test_samples = split_train_test(self.root, lambda path:
                (sample_above_confidence_threshold(path, self.signer_confidence_threshold) if self.signer_confidence_threshold != 0 else True))
            store_samples_to_csv(train_path, self.train_samples)
            store_samples_to_csv(test_path, self.test_samples)
        self.max_label_len = max(map(len, _yield_tokens(self.train_samples + self.test_samples, self.tokenizer)))

        special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.vocab = build_vocab_from_iterator(_yield_tokens(self.train_samples, self.tokenizer),
                                                            min_freq = words_min_freq,
                                                            specials = special_symbols,
                                                            special_first = True)
        # by default returns <unk> index
        self.vocab.set_default_index(0)
        

    def __len__(self) -> int:
        return len(self.train_samples if self.mode == "train" else self.test_samples)

    def __getitem__(self, index: int) -> Sample:
        paths = get_cut_paths((self.train_samples if self.mode == "train" else self.test_samples)[index])
        with paths['signer'].open() as signer_file:
            signer: SignerData = json.load(signer_file)
        with paths['json'].open() as data_file:
            data: CutData = json.load(data_file)
        clip = (
            None if not self.load_clips
            else _load_clip_as_tensors(paths) if self.clip_transform is None
            else self.clip_transform(_load_clip_as_tensors(paths))
        )
        keypoints = (
            None if not self.load_keypoints
            else signer['keypoints'] if self.keypoints_transform is None
            else self.keypoints_transform(signer['keypoints'])
        )
        label = data['label'] if self.label_transform is None else self.label_transform(data['label'])
        return (clip, keypoints, label)
    
    def __iter__(self) -> Iterator[Sample]:
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def get_token_idx(self, token: str) -> int:
        return self.vocab.__getitem__(token)
