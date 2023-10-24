from pathlib import Path
import pandas as pd
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip


def main():
	labels = pd.read_csv('lsat/data/labels.csv')
	out_path = Path('lsat/data/cuts')
	out_path.mkdir(exist_ok=True,parents=True)

	video = labels.at[0, 'video']
	video_file = VideoFileClip(f"lsat/data/raw/{labels.at[0, 'playlist']}/{video}.mp4")
	for idx, row in labels.iterrows():
		if video != row['video']:
			video = row['video']
			video_file = VideoFileClip(f"lsat/data/raw/{row['playlist']}/{video}.mp4")
		print(f"Clip {idx}/{len(labels) - 1}")
		if not (out_path / f"{idx}.mp4").exists():
			prev_delta = row['prev_delta'] if (row['prev_delta'] != np.nan and row['prev_delta'] <= .5) else .5
			post_delta = row['post_delta'] if (row['post_delta'] != np.nan and row['post_delta'] <= .5) else .5
			newvid = video_file.subclip(row['start'] - prev_delta, row['end'] + post_delta)
			newvid.write_videofile(str(out_path / f"{row['id']}.mp4"), audio=False, codec="libx264", fps=video_file.fps)

if __name__ == "__main__":
    main()
