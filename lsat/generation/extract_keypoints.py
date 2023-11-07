import os
from typing import Optional

import cv2
import h5py
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO
from mediapipe import solutions
import torch

from helpers import load_video

from hints.Frame import Frame


JOINTS_SIZE = np.float16

def process_keys(frame_keypoints, box: NDArray[JOINTS_SIZE]) -> NDArray[JOINTS_SIZE]:
	pose = [coord for landmark in frame_keypoints.pose_landmarks.landmark for coord in [landmark.x*(box[2]-box[0])+box[0], landmark.y*(box[3]-box[1])+box[1], None, landmark.visibility]] if frame_keypoints.pose_landmarks is not None else [None]*33*4
	face = [coord for landmark in frame_keypoints.face_landmarks.landmark for coord in [landmark.x*(box[2]-box[0])+box[0], landmark.y*(box[3]-box[1])+box[1], landmark.z, None]] if frame_keypoints.face_landmarks is not None else [None]*468*4
	rhand = [coord for landmark in frame_keypoints.right_hand_landmarks.landmark for coord in [landmark.x*(box[2]-box[0])+box[0], landmark.y*(box[3]-box[1])+box[1], landmark.z, None]] if frame_keypoints.right_hand_landmarks is not None else [None]*21*4
	lhand = [coord for landmark in frame_keypoints.left_hand_landmarks.landmark for coord in [landmark.x*(box[2]-box[0])+box[0], landmark.y*(box[3]-box[1])+box[1], landmark.z, None]] if frame_keypoints.left_hand_landmarks is not None else [None]*21*4
	return np.array(pose + face + rhand + lhand, dtype=JOINTS_SIZE)

def run_holistic(frames: NDArray[np.uint8], box: NDArray[JOINTS_SIZE]) -> NDArray[JOINTS_SIZE]:
	keypoints = np.empty((len(frames), 33*4+468*4+21*4+21*4), dtype=JOINTS_SIZE)
	keypoints.fill(np.nan)
	with solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic: # type: ignore
		for i_frame, frame in enumerate(frames):
			keypoints[i_frame] = process_keys(holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), box)
	return keypoints

def crop_person(box: NDArray[JOINTS_SIZE], frame: Frame) -> Frame:
	bottom = coord_to_pixel(box[1])
	top = coord_to_pixel(box[3])
	left = coord_to_pixel(box[0])
	right = coord_to_pixel(box[2])
	fr = frame.copy()[bottom:top, left:right, :]
	return fr

def blackout_box(box: NDArray[JOINTS_SIZE], frame: Frame) -> Frame:
	bottom = coord_to_pixel(box[1])
	top = coord_to_pixel(box[3])
	left = coord_to_pixel(box[0])
	right = coord_to_pixel(box[2])
	fr = frame.copy()
	fr[top:, :, :] = (0,0,0)
	fr[:bottom, :, :] = (0,0,0)
	fr[:, :left, :] = (0,0,0)
	fr[:, right:, :] = (0,0,0)
	return fr

def get_shared_area(box1: NDArray[JOINTS_SIZE], box2: NDArray[JOINTS_SIZE]) -> float:
	'Returns the area shared by two boxes in a range from 0 to 1'
	x1 = max(box1[0], box2[0])
	y1 = max(box1[1], box2[1])
	x2 = min(box1[2], box2[2])
	y2 = min(box1[3], box2[3])
	shared_area = max(0, x2 - x1) * max(0, y2 - y1)
	return shared_area / ((box1[2] - box1[0]) * (box1[3] - box1[1]))

def last_valid_box(boxes: NDArray[JOINTS_SIZE]) -> Optional[NDArray[JOINTS_SIZE]]:
	'Returns the last row that does not contain a nan box'
	for box in boxes[::-1]:
		if not np.isnan(box).any():
			return box
	return None

coord_to_pixel = lambda x : int(x) if not np.isnan(x) else 0

def main():
	input_path = "lsat/data/cuts"
	keypoints_path = "lsat/data/keypoints.h5"
	clips = sorted(os.listdir(input_path))

	# remove clips that have already been processed if keypoints file exists
	if os.path.exists(keypoints_path):
		with h5py.File(keypoints_path, 'r') as hdf5_file:
			clips = [clip for clip in clips if clip not in list(hdf5_file.keys())]

	model = YOLO("yolov8n-pose.pt")

	np.random.seed(0)
	torch.manual_seed(0)

	for i_person, clip in list(enumerate(clips)):
		print(f"Processing clip {i_person+1}/{len(clips)}: {clip}")
		try:
			video, _, frame_count = load_video(f"{input_path}/{clip}")
			
			print(f"Loaded {len(video)} frames")

			# track people in video
			print("Tracking people in video")
			results = model.track(source=f"{input_path}/{clip}", persist=True, conf=0.75, verbose=False, stream=True)
			
			# get bounding box for each person
			print("Getting bounding boxes for each person")
			people_frame_level_boxes: list[NDArray[JOINTS_SIZE]] = []
			last_boxes: list[NDArray[JOINTS_SIZE]] = []
			for i_frame, frame in enumerate(results):
				# boxes are in format x1, y1, x2, y2

				people_frame_boxes = [np.array(box)[:4] for box in frame.boxes.data.tolist()]
				# check if any value of the boxes is larger than 3000, if so print the box
				if any([any([coord > 3000 for coord in box]) for box in people_frame_boxes]):
					print(f"Box larger than 3000 in frame {i_frame+1} of clip {clip}")
					print(people_frame_boxes)

				# append box to boxes in the index where the last box shares the most area with the current box, if no box shares more than 50% of area, append to the end
				for i_person, frame_box in enumerate(people_frame_boxes):
					shared_areas = [get_shared_area(frame_box, last_person_box) for last_person_box in last_boxes]
					if any(map(lambda x: x>.5, shared_areas)):
						max_shared_area = max(shared_areas)
						people_frame_level_boxes[shared_areas.index(max_shared_area)][i_frame] = people_frame_boxes[i_person]
						last_boxes[shared_areas.index(max_shared_area)] = people_frame_boxes[i_person]
					else:
						new_box = np.empty((frame_count, 4), dtype=JOINTS_SIZE)
						new_box.fill(np.nan)
						people_frame_level_boxes.append(new_box)
						people_frame_level_boxes[-1][i_frame] = people_frame_boxes[i_person]
						last_boxes.append(people_frame_boxes[i_person])

			# get clip wise bounding boxes for each person
			clip_level_boxes: list[NDArray[JOINTS_SIZE]] = []
			for i_person, person_box in enumerate(people_frame_level_boxes):
				clip_level_boxes.append(np.stack([np.nanmin(person_box[:, 0]), np.nanmin(person_box[:, 1]), np.nanmax(person_box[:, 2]), np.nanmax(person_box[:, 3])]))
			
			# people_keypoints contains for each person, a matrix of frames x keypoints
			people_keypoints: list[NDArray[JOINTS_SIZE]] = []
			for i_person, person_box in enumerate(clip_level_boxes):
				print(f"Processing signer {i_person+1}/{len(clip_level_boxes)}")
				signer_video = np.empty((len(video), coord_to_pixel(person_box[3])-coord_to_pixel(person_box[1]), coord_to_pixel(person_box[2])-coord_to_pixel(person_box[0]), 3), dtype=np.uint8)
				signer_video.fill(0)
				for i_frame, (frame, roi) in enumerate(zip(video, people_frame_level_boxes[i_person])):
					signer_video[i_frame] = crop_person(person_box, blackout_box(roi, frame))
				signer_keypoints = run_holistic(signer_video, person_box)
				people_keypoints.append(signer_keypoints)
		
			with h5py.File(keypoints_path, 'a') as hdf5_file:
				clip_group = hdf5_file.create_group(clip)
				for i_person, (person_keypoints, person_box) in enumerate(zip(people_keypoints, people_frame_level_boxes)):
					signer_group = clip_group.create_group(f"signer_{i_person}")
					signer_group.create_dataset('keypoints', data=person_keypoints)
					signer_group.create_dataset('boxes', data=person_box)
		
		except Exception as e:
			print(f"Failed to process clip {clip}: {e}")


if __name__ == "__main__":
	main()