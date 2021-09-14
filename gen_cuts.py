from typing import TextIO
import os, sys, json
from moviepy.video.io.VideoFileClip import VideoFileClip


input = 'raw/'
# Se descarta un video con los subtitulos descoordinados
excluded = ['Noticias en Lengua de Señas Argentina (resumen semanal 06_12_2020)-d7akwvWNPrU.es-419.vtt']
sub_files = [f for f in os.listdir(input) if f.endswith('.vtt') if f not in excluded]

def str_to_secs(time: str) -> float:
    'Amount of seconds for string in xx:xx:xx format'
    hours, mins, secs = time.split(':')
    return float(hours)*3600 + float(mins)*60 + float(secs)

SubsList = list[tuple[float, float, str]]
def process_sub_file(file: TextIO) -> SubsList:
    subs: SubsList = []
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
                if sub is not None and sub[:-1] != "[Música]":
                    subs.append((start, end, sub[:-1].replace('- ', '').lower()))
                start = end = sub = None
    return subs

if not os.path.isdir('data'):
    os.mkdir('data')
if not os.path.isdir('data/cuts'):
    os.mkdir('data/cuts')

for filename in sub_files:
    name = filename[:-11]
    outdir = "data/cuts/{}/".format(name)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    with open(input + filename, 'r', encoding='utf-8') as subs_file:
        subs = process_sub_file(subs_file)
    with VideoFileClip("raw/{}.mp4".format(name)) as video:
        for i, (start, end, sub) in enumerate(subs):
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
        os.remove("raw/{}.es-419.vtt".format(name))
