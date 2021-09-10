import json
import os
import sys
from moviepy.video.io.VideoFileClip import VideoFileClip
# La opción usando esta función es (mucho) más rápida pero deja framse congelados al final
#from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


with open("data/subs_dict.json", 'r', encoding='utf-8') as subs_dict_file:
  subs_dict: dict[str, list[tuple[float, float, str]]] = json.load(subs_dict_file)

# Se descarta un video con los subtitulos descoordinados
excluded = ['Noticias en Lengua de Señas Argentina (resumen semanal 06_12_2020)-d7akwvWNPrU']

for name, subs in subs_dict.items():
    if name not in excluded:
        dir = "data/cuts/{}/".format(name)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        with VideoFileClip("raw/{}.mp4".format(name)) as video:
            for i, (start, end, sub) in enumerate(subs):
                if not os.path.isfile((dir + str(i) + ".mp4")):
                    new = video.subclip(start, end)
                    new.write_videofile((dir + str(i) + ".mp4"), audio=False)
            #ffmpeg_extract_subclip("raw/{}.mp4".format(name), start, end, targetname=(dir + str(i) + ".mp4"))
    if len(sys.argv) > 1 and sys.argv[1] == "-d":
        os.remove("raw/{}.mp4".format(name))
        os.remove("raw/{}.es-419.vtt".format(name))

