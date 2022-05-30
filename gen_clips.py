import argparse, json
from typing import Optional, TextIO
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip

Subt = tuple[float, float, str]

def str_to_secs(time: str) -> float:
    'Amount of seconds for string in xx:xx:xx format'
    hours, mins, secs = time.replace(',','.').split(':')
    return float(hours)*3600 + float(mins)*60 + float(secs)

def process_sub_file(file: TextIO, illegal_subs: list[str]) -> list[Subt]:
    'Takes a .vtt file and returns a list of Subt'
    subs: list[Subt] = []
    start = end = None
    sub: Optional[str] = None
    for line in file:
        if ' --> ' in line:
            start, end = str_to_secs(line.split(' --> ')[0]), str_to_secs(line.split(' --> ')[1][:-1])
        elif start is not None and end is not None:
            if line != '\n':
                if sub is not None:
                    sub = sub[:-1] + ' ' + line
                else:
                    sub = line
            else:
                if sub is not None and sub[:-1] not in illegal_subs:
                    subs.append((start, end, sub[:-1].replace('- ', '').lower()))
                start = end = sub = None
    if sub is not None and sub[:-1] not in illegal_subs and (start is not None and end is not None):
        subs.append((start, end, sub[:-1].replace('- ', '').lower()))
    return subs


def main():
    'Parses subtitles files (.vtt) and generates a clip for each piece of subtitles'
    parser = argparse.ArgumentParser(description='''Parses subtitles files (.vtt) and generates, for each line of subtitles of the videos the clip corresponding to the that line of subtitles and a json file with it's metadata.''')
    parser.add_argument('--delete', '-d', help='deletes both video and subtitle file after processing', action='store_true')
    must_del: bool = parser.parse_args().delete

    input_path = Path('data/raw')
    cuts_path = Path('data/cuts')
    cuts_path.mkdir(exist_ok=True,parents=True)

    # Se descarta un video con los subtitulos descoordinados
    excluded = ['noticias-en-lengua-de-senas-argentina-resumen-semanal-06122020.vtt']
    sub_files = [f for f in input_path.glob('**/*.vtt') if f.name not in excluded]
    illegal_subs = ["[Música", "[♪ música ♪]", "[Música]", "[\u266a m\u00fasica \u266a]", "[m\u00fasica]"]

    for vid_idx, filename in enumerate(sub_files):
        name = f"{filename.parent.name}/{filename.name[:-4]}"
        outdir = Path(cuts_path / f"{name}")
        outdir.mkdir(exist_ok=True,parents=True)
        with filename.open(encoding='utf-8') as subs_file:
            subs = process_sub_file(subs_file, illegal_subs)
        with VideoFileClip(str(filename)[:-3] + "mp4") as video:
            for i, (start, end, sub) in enumerate(subs):
                print(f"Video {vid_idx + 1}/{len(sub_files)} - Clip {i + 1}/{len(subs)}")
                if not (outdir / f"{i}.json").exists():
                    newvid = video.subclip(start, end)
                    newvid.write_videofile(str(outdir / f"{i}.mp4"), audio=False, codec="libx264")
                    with (outdir / f"{i}.json").open(mode='w', encoding='utf-8') as data_file:
                        json.dump({
                            'label': sub,
                            'start': start,
                            'end': end,
                            'video': filename.name[:-4],
                            'playlist': filename.parent.name
                        }, data_file, indent=4)
        if must_del:
            (input_path / f"{name}.mp4").unlink()
            (input_path / f"{name}.vtt").unlink()

if __name__ == "__main__":
    main()
