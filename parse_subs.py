import os
import json


dir = 'raw/'
sub_files = [f for f in os.listdir(dir) if f.endswith('.vtt')]

def str_to_secs(time: str) -> float:
    'Amount of seconds for string in xx:xx:xx format'
    hours, mins, secs = time.split(':')
    return float(hours)*3600 + float(mins)*60 + float(secs)

# subs_dict contains a dictionary with the form {videoTitle: [(start, end, line)]} where start and end are the time in seconds of line
subs_dict: dict[str, list[tuple[float, float, str]]] = {}

for filename in sub_files:
    name = filename[:-11]
    subs_dict[name] = []
    with open(dir + filename, 'r', encoding='utf-8') as file:
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
                        subs_dict[name].append((start, end, sub[:-1].replace('- ', '').lower()))
                    start = end = sub = None

with open("data/subs_dict.json", 'w', encoding='utf-8') as subs_dict_file:
  json.dump(subs_dict, subs_dict_file)
