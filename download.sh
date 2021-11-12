#!/bin/bash

cd raw

# Playlists de resumen semanal, ecologia, ultimo momento y #leyfederalLSA
declare -a playlists=("https://www.youtube.com/playlist?list=PLhysX0rYXWV2xM3T4KAqaAEg-GEhDXwai"
                        "https://www.youtube.com/playlist?list=PLhysX0rYXWV2DioiHvMjzJs_4UWifmPo9"
                        "https://www.youtube.com/playlist?list=PLhysX0rYXWV2wQnK2nxU4gvcmq5yrHli3"
                        "https://www.youtube.com/playlist?list=PLhysX0rYXWV2WNLgIyBiyn3wizILOTuP6"
                    )

for p in "${playlists[@]}"; do
    youtube-dl --download-archive downloaded.txt --write-sub --sub-lang es-419 --no-post-overwrites -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]' $p
done;
