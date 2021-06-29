#!/bin/sh

youtube-dl --download-archive downloaded.txt --write-sub --sub-lang es-419 --no-post-overwrites $1
