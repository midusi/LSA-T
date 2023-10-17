import os

import cv2
import h5py
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO
import mediapipe as mp

from helpers import load_video

from hints.Box import Box, format_box
from hints.Frame import Frame


KEYPOINTS_SIZE = np.float16

def process_keys(frame_keypoints, box: Box) -> NDArray[KEYPOINTS_SIZE]:
	pose = [coord for landmark in frame_keypoints.pose_landmarks.landmark for coord in [landmark.x*(box['x2']-box['x1'])+box['x1'], landmark.y*(box['y2']-box['y1'])+box['y1'], None, landmark.visibility]] if frame_keypoints.pose_landmarks is not None else [None]*33*4
	face = [coord for landmark in frame_keypoints.face_landmarks.landmark for coord in [landmark.x*(box['x2']-box['x1'])+box['x1'], landmark.y*(box['y2']-box['y1'])+box['y1'], landmark.z, None]] if frame_keypoints.face_landmarks is not None else [None]*468*4
	rhand = [coord for landmark in frame_keypoints.right_hand_landmarks.landmark for coord in [landmark.x*(box['x2']-box['x1'])+box['x1'], landmark.y*(box['y2']-box['y1'])+box['y1'], landmark.z, None]] if frame_keypoints.right_hand_landmarks is not None else [None]*21*4
	lhand = [coord for landmark in frame_keypoints.left_hand_landmarks.landmark for coord in [landmark.x*(box['x2']-box['x1'])+box['x1'], landmark.y*(box['y2']-box['y1'])+box['y1'], landmark.z, None]] if frame_keypoints.left_hand_landmarks is not None else [None]*21*4
	return np.array(pose + face + rhand + lhand, dtype=KEYPOINTS_SIZE)

def run_holistic(frames: NDArray[np.uint8], box: Box) -> NDArray[KEYPOINTS_SIZE]:
	keypoints = np.empty((len(frames), 33*4+468*4+21*4+21*4), dtype=KEYPOINTS_SIZE)
	with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
		for i_frame, frame in enumerate(frames):
			keypoints[i_frame] = process_keys(holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), box)
	return keypoints


def crop_person(box: Box, frame: Frame) -> Frame:
	bottom = int(box['y1'])
	top = int(box['y2'])
	left = int(box['x1'])
	right = int(box['x2'])
	fr = frame.copy()[bottom:top, left:right, :]
	return fr

def blackout_box(box: Box, frame: Frame) -> Frame:
	bottom = int(box['y1'])
	top = int(box['y2'])
	left = int(box['x1'])
	right = int(box['x2'])
	fr = frame.copy()
	fr[top:, :, :] = (0,0,0)
	fr[:bottom, :, :] = (0,0,0)
	fr[:, :left, :] = (0,0,0)
	fr[:, right:, :] = (0,0,0)
	return fr

def get_shared_area(box1: Box, box2: Box) -> float:
	'Returns the area shared by two boxes in a range from 0 to 1'
	x1 = max(box1['x1'], box2['x1'])
	y1 = max(box1['y1'], box2['y1'])
	x2 = min(box1['x2'], box2['x2'])
	y2 = min(box1['y2'], box2['y2'])
	shared_area = max(0, x2 - x1) * max(0, y2 - y1)
	return shared_area / ((box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1']))

def last_valid_box(boxes: list[Box]) -> Box:
	'Returns the last valid box in a list of boxes'
	for box in reversed(boxes):
		if box['x1'] != 0 and box['y1'] != 0 and box['x2'] != 0 and box['y2'] != 0:
			return box
	return format_box([0]*4)

def main():
	input_path = "../data/cuts"
	keypoints_path = "../data/keypoints.h5"
	clips = sorted(os.listdir(input_path))

	# remove clips that have already been processed if keypoints file exists
	if os.path.exists(keypoints_path):
		with h5py.File(keypoints_path, 'r') as hdf5_file:
			clips = [clip for clip in clips if clip not in list(hdf5_file.keys())]

	model = YOLO("yolov8n-pose.pt")

	for i_person, clip in list(enumerate(clips)):
		try:
			print(f"Processing clip {i_person+1}/{len(clips)}: {clip}")
			video, _ = load_video(f"{input_path}/{clip}")
			
			print(f"Loaded {len(video)} frames")

			# track people in video
			print("Tracking people in video")
			results = model.track(source=f"{input_path}/{clip}", persist=True, conf=0.75, verbose=False, stream=True)
			
			# get bounding box for each person
			print("Getting bounding boxes for each person")
			people_frame_level_boxes: list[list[Box]] = []
			for i_frame, frame in enumerate(results):
				people_frame_boxes = [format_box(box) for box in frame.boxes.data.tolist()]
				# append box to boxes in the index where the last box shares the most area with the current box, if no box shares more than 50% of area, append to the end
				for i_person, person_boxes in enumerate(people_frame_level_boxes):
					if len(people_frame_boxes) > 0 and len(person_boxes) > 0:
						# get the box with the most area shared with the last box in the roi
						shared_areas = [get_shared_area(frame_box, last_valid_box(person_boxes)) for frame_box in people_frame_boxes]
						max_shared_area = max(shared_areas)
						if max_shared_area > 0.5:
							people_frame_level_boxes[i_person].append(people_frame_boxes.pop(shared_areas.index(max_shared_area)))
					elif len(person_boxes) == 0:
						people_frame_level_boxes[i_person].append(people_frame_boxes.pop(0))
				# add any remaining boxes to the end filling with empty boxes before
				for frame_box in people_frame_boxes:
					current_max_len = max([len(boxes) for boxes in people_frame_level_boxes]) if len(people_frame_level_boxes) > 0 else 0
					people_frame_level_boxes.append([format_box([0]*4)]*(current_max_len-1)+[frame_box])
				# if there are people that have not been detected in this frame, add empty boxes to the end
				for i_person, person_boxes in enumerate(people_frame_level_boxes):
					if len(person_boxes) < max([len(boxes) for boxes in people_frame_level_boxes]):
						people_frame_level_boxes[i_person].append(format_box([0]*4))

			# get clip wise bounding boxes for each person
			clip_level_boxes: list[Box] = []
			for i_signer, signer_box in enumerate(people_frame_level_boxes):
				for frame_box in signer_box:
					if i_person < len(clip_level_boxes):
						clip_level_boxes[i_signer]['x1'] = min(clip_level_boxes[i_signer]['x1'], frame_box['x1'])
						clip_level_boxes[i_signer]['y1'] = min(clip_level_boxes[i_signer]['y1'], frame_box['y1'])
						clip_level_boxes[i_signer]['x2'] = max(clip_level_boxes[i_signer]['x2'], frame_box['x2'])
						clip_level_boxes[i_signer]['y2'] = max(clip_level_boxes[i_signer]['y2'], frame_box['y2'])
					else:
						clip_level_boxes.append(frame_box)
			
			# people_keypoints contains for each person, a matrix of frames x keypoints
			people_keypoints: list[NDArray[KEYPOINTS_SIZE]] = []
			for i_person, person_boxes in enumerate(clip_level_boxes):
				print(f"Processing signer {i_person+1}/{len(clip_level_boxes)}")
				signer_video = np.empty((len(video), int(person_boxes['y2'])-int(person_boxes['y1']), int(person_boxes['x2'])-int(person_boxes['x1']), 3), dtype=np.uint8)
				for i_frame, (frame, roi) in enumerate(zip(video, people_frame_level_boxes[i_person])):
					signer_video[i_frame] = crop_person(person_boxes, blackout_box(roi, frame))
				signer_keypoints = run_holistic(signer_video, person_boxes)
				people_keypoints.append(signer_keypoints)
		
			with h5py.File(keypoints_path, 'a') as hdf5_file:
				clip_group = hdf5_file.create_group(clip)
				for i_signer, (person_keypoints, person_boxes) in enumerate(zip(people_keypoints, people_frame_level_boxes)):
					signer_group = clip_group.create_group(f"signer_{i_signer}")
					signer_group.create_dataset('keypoints', data=person_keypoints)
					signer_group.create_dataset('boxes', data=np.stack([[box['x1'], box['y1'], box['x2'], box['y2']] for box in person_boxes]))
				
		except Exception as e:
			print(f"Error processing clip {clip}: {e}")
			continue


if __name__ == "__main__":
	main()