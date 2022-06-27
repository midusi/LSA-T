from pathlib import Path

def get_cut_paths(cut: Path) -> dict[str, Path]:
    'Return paths for al the files (if exists) corresponding to a single clip'
    name = cut.name[:-4]
    return {
        'mp4': cut,
        'json': cut.parent / f"{name}.json",
        'signer': cut.parent / f"{name}_signer.json",
        'ap': cut.parent / f"{name}_ap.json",
        'ap_raw': cut.parent / name / "alphapose-results.json",
    }
