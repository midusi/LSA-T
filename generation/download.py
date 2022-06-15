from pathlib import Path
from pytube import Playlist
from pytube.cli import on_progress

from helpers.slugify import slugify

def main():
    'Downloads videos and subtitles into raw folder.'
    # Playlists de resumen semanal, ecologia, ultimo momento y #leyfederalLSA
    playlists = {
        'resumen_semanal': "https://www.youtube.com/playlist?list=PLhysX0rYXWV2xM3T4KAqaAEg-GEhDXwai",
        'ecologia': "https://www.youtube.com/playlist?list=PLhysX0rYXWV2DioiHvMjzJs_4UWifmPo9",
        'ultimo_momento': "https://www.youtube.com/playlist?list=PLhysX0rYXWV2wQnK2nxU4gvcmq5yrHli3",
        'ley_federal_lsa': "https://www.youtube.com/playlist?list=PLhysX0rYXWV2WNLgIyBiyn3wizILOTuP6"
    }
    path = Path("data/raw")
    path.mkdir(exist_ok=True,parents=True)

    print("Fetching video list")
    videos = [(vid_path, yt) for pl_videos in map(lambda pl: 
                [((path / f"{pl[0]}/{slugify(video.title)}.vtt"), video) for video in Playlist(pl[1]).videos],
                playlists.items())
            for (vid_path, yt) in pl_videos if ('es-419' in yt.captions) and not vid_path.exists()]

    for idx, (vid_path, yt) in enumerate(videos):
        vid_path.parent.mkdir(exist_ok=True,parents=True)
        yt.register_on_progress_callback(on_progress)
        st = yt.streams.filter(adaptive=True, file_extension='mp4').order_by('resolution').last()
        print(f"\nVideo {idx + 1}/{len(videos)}: {yt.title.replace('/', '-')}\n{st}")
        if st is not None:
            st.download(output_path=path.resolve(), filename=str(vid_path.absolute()).replace('vtt','mp4'))
            with vid_path.open(mode='w') as subs_file:
                subs_file.write(yt.captions['es-419'].generate_srt_captions())

if __name__ == "__main__":
    main()
