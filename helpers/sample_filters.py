import json
from pathlib import Path
from typing import Callable

from torchtext.vocab import Vocab

from type_hints import CutData
from helpers.get_cut_paths import get_cut_paths
from helpers.get_score import get_score

def sample_contains_oov(data_path: Path, vocab: Vocab, tokenizer: Callable[[str], list[str]]) -> bool:
    with open(data_path) as data_file:
        data: CutData = json.load(data_file)
        return not all(map(vocab.__contains__, tokenizer(data['label'])))

def sample_above_confidence_threshold(data_path: Path, threshold: float) -> bool:
    with open(get_cut_paths(data_path)['signer']) as signer_file:
        line = ""
        while ']' not in line:
            line += signer_file.readline()
        line = line[:-2] + '}'
        signer_data = json.loads(line)
        return get_score(signer_data['scores']) >= threshold
