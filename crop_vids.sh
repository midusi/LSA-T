#!/bin/bash

for f in res/*; do
    id=${f#*/}
    arr=()
    while read line; do
        arr+=($line)
    done < "$f/box.txt"
    echo "aasd${arr[0]}, ${arr[1]}, ${arr[2]}, ${arr[3]}"
    ffmpeg -y -i "$f/AlphaPose_$id.mp4" -filter:v "crop=${arr[2]}:${arr[3]}:${arr[0]}:${arr[1]}" "$f/crop.mp4"
done