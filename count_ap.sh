#!/bin/bash

IFS=$'\n'; set -f;
total=0
missing=0

for vid in $(find 'data/cuts' -name '*.mp4'); do
	if [ ! -f "${vid::-4}/alphapose-results.json" ]; then
		missing=$((missing+1))
	fi;
	total=$((total+1))
done;

echo $total
echo $missing

unset IFS; set +f;
