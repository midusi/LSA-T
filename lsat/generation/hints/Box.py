from typing import TypedDict


class Box(TypedDict):
	'Standard box data format for ROI'
	x1: float
	y1: float
	x2: float
	y2: float

def format_box(box: list[float]) -> Box:
	'Creates box from list of strings'
	return {
		'x1': round(box[0], 2),
		'y1': round(box[1], 2),
		'x2': round(box[2], 2),
		'y2': round(box[3], 2)
	}
