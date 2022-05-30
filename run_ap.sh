#!/bin/bash

if [ $# -gt 0 ]; then
	ap_loc=$1
else
	echo "AlphaPose path missing"
	exit 1
fi;
path=$PWD

IFS=$'\n'; set -f;

i=0
for vid in $(find 'data/cuts' -name '*.mp4'); do
	if [ ! -f "${vid::-4}/alphapose-results.json" ]; then
		echo $i
		vid_path=$(realpath $vid)
		echo $vid_path
		cd $ap_loc
		echo $PWD
		python "scripts/demo_inference.py" --cfg "configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml" --checkpoint "pretrained_models/halpe136_fast_res50_256x192.pth" --qsize 4096 --video "$vid_path" --outdir "${vid_path::-4}/" --pose_track;
		cd $path
	fi;
	i=$((i+1))
done;

unset IFS; set +f;
