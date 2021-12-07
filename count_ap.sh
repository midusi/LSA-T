#!/bin/bash

IFS=$'\n'; set -f;
total=0
completed=0

for vid in $(find 'data/cuts' -name '*.mp4'); do
	if [ -f "${vid::-4}/alphapose-results.json" ]; then
		completed=$((completed+1))
	fi;
	total=$((total+1))
done;

echo "$completed / $total"

unset IFS; set +f;
