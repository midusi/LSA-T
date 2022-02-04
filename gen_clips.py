from typing import TextIO
import os, sys, json
from moviepy.video.io.VideoFileClip import VideoFileClip


Subt = tuple[float, float, str]

input = 'raw/'
# Se descarta un video con los subtitulos descoordinados
excluded = ['noticias-en-lengua-de-senas-argentina-resumen-semanal-06122020.vtt']
sub_files = [f for f in os.listdir(input) if f.endswith('.vtt') if f not in excluded]

illegal_subs = ["[Música", "[♪ música ♪]", "[Música]"]

def str_to_secs(time: str) -> float:
    'Amount of seconds for string in xx:xx:xx format'
    hours, mins, secs = time.replace(',','.').split(':')
    return float(hours)*3600 + float(mins)*60 + float(secs)

def process_sub_file(file: TextIO) -> list[Subt]:
    subs: list[Subt] = []
    start = end = sub = None
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
    if sub is not None and sub[:-1] not in illegal_subs:
        subs.append((start, end, sub[:-1].replace('- ', '').lower()))
    return subs

def main():
    if not os.path.isdir('data'):
        os.mkdir('data')
    if not os.path.isdir('data/cuts'):
        os.mkdir('data/cuts')

    for vid_idx, filename in enumerate(sub_files):
        name = filename[:-4]
        outdir = "data/cuts/{}/".format(name)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        with open(input + filename, 'r', encoding='utf-8') as subs_file:
            subs = process_sub_file(subs_file)
        with VideoFileClip("raw/{}.mp4".format(name)) as video:
            for i, (start, end, sub) in enumerate(subs):
                print("Video {}/{} - Clip {}/{}".format(vid_idx + 1, len(sub_files), i + 1, len(subs)))
                if not os.path.isfile((outdir + str(i) + ".json")):
                    newvid = video.subclip(start, end)
                    newvid.write_videofile((outdir + str(i) + ".mp4"), audio=False)
                    with open(outdir + str(i) + ".json", 'w', encoding='utf-8') as data_file:
                        json.dump({
                            'label': sub,
                            'start': start,
                            'end': end,
                            'video': name
                        }, data_file)
        if len(sys.argv) > 1 and sys.argv[1] == "-d":
            os.remove("raw/{}.mp4".format(name))
            os.remove("raw/{}.vtt".format(name))

if __name__ == "__main__":
    main()