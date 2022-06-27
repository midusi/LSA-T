import json
from pathlib import Path
from moviepy.editor import VideoFileClip
from typing import Callable
from numpy.typing import ArrayLike

from lsat.helpers.get_cut_paths import get_cut_paths
from lsat.helpers.group_kds import group_kds
from lsat.typing import KeypointData, SignerData, Box


def draw_rectangle(box: Box) -> Callable[[ArrayLike], ArrayLike]:
    def draw(frame: ArrayLike) -> ArrayLike:
        '''Draw a rectangle in the frame'''
        fr = frame.copy()
        bottom = int(box['y1'])
        top = int(bottom + box['height'])
        left = int(box['x1'])
        right = int(left + box['width'])
        fr[top-5:top+5, left:right] = 255 - fr[top-5:top+5, left:right]
        fr[bottom-5:bottom+5, left:right] = 255 - fr[bottom-5:bottom+5, left:right]
        fr[bottom:top, left-5:left+5] = 255 - fr[bottom:top, left-5:left+5]
        fr[bottom:top, right-5:right+5] = 255 - fr[bottom:top, right-5:right+5]
        return fr
    return draw

def draw_keypoints(
        keypoints: list[KeypointData],
        fps: float,
        size: int = 5,
        threshold: float = 0) -> Callable[[Callable[[float], ArrayLike], float], ArrayLike]:
    def draw(get_frame: Callable[[float], ArrayLike], t: float) -> ArrayLike:
        '''Draw keypoints in the frame'''
        fr = get_frame(t).copy()
        keypoints_t: list[float] = keypoints[min(int(t * fps), len(keypoints)-1)]['keypoints']
        it = iter(keypoints_t)
        for x, y, conf in list(zip(it, it, it))[94:]:
            if conf > threshold:
                fr[int(y)-size:int(y)+size:,int(x)-size:int(x)+size] = 255-fr[int(y)-size:int(y)+size:,int(x)-size:int(x)+size]
        return fr
    return draw

def gen_vis_db():
    'Generates a lightweight database with videos in lower quality that has keipoints and roi embebbed on them.'
    source = Path('./data/cuts')
    out = Path('./data/cuts_visualization')
    out.mkdir(exist_ok=True,parents=True)
    cuts = source.glob('**/*.mp4')
    for i, cut in enumerate(cuts):
        print(f"Video {i}")
        outpath = out / cut.parent.name
        outpath.mkdir(exist_ok=True,parents=True)
        clip_files = get_cut_paths(cut)
        out_clip = VideoFileClip(str(clip_files['mp4']))
        with clip_files['signer'].open() as signerf:
            signer: SignerData = json.load(signerf)
        with clip_files['ap'].open() as apf:
            ap: list[KeypointData] = json.load(apf)
        signers = group_kds(ap)
        for s in signers:
            out_clip: VideoFileClip = out_clip.fl(draw_keypoints(s, out_clip.fps,  7, 0.5))
        out_clip = out_clip.fl_image(draw_rectangle(signer['roi']))
        out_clip = out_clip.resize(height = 240)
        out_clip.write_videofile(str((outpath / cut.name)), codec="libx264", fps=10, verbose=False, logger=None)
        with open(clip_files['json']) as dataf:
            data = json.load(dataf)
            data['scores'] = signer['scores']
            data['roi'] = signer['roi']
        with (outpath / f"{cut.name[:-4]}.json").resolve().open('w') as out_dataf:
            json.dump(data, out_dataf, indent=4)


if __name__ == "__main__":
    gen_vis_db()
