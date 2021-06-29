import json
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


with open("data/subs_dict.json", 'r', encoding='utf-8') as subs_dict_file:
  subs_dict: dict[str, list[tuple[float, float, str]]] = json.load(subs_dict_file)

for name, subs in subs_dict.items():
    dir = "data/cuts/{}/".format(name)
    if not os.path.isdir(dir):
        os.mkdir(dir)
    for i, (start, end, sub) in enumerate(subs):
        ffmpeg_extract_subclip("raw/{}.mp4".format(name), start, end, targetname=(dir + str(i) + ".mp4"))
