#!/bin/sh

cd raw
youtube-dl --download-archive downloaded.txt --write-sub --sub-lang es-419 --no-post-overwrites -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]' $1
