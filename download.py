from pytube import Playlist, YouTube
from pytube.cli import on_progress


# Playlists de resumen semanal, ecologia, ultimo momento y #leyfederalLSA
playlists = ["https://www.youtube.com/playlist?list=PLhysX0rYXWV2xM3T4KAqaAEg-GEhDXwai",
            "https://www.youtube.com/playlist?list=PLhysX0rYXWV2DioiHvMjzJs_4UWifmPo9",
            "https://www.youtube.com/playlist?list=PLhysX0rYXWV2wQnK2nxU4gvcmq5yrHli3",
            "https://www.youtube.com/playlist?list=PLhysX0rYXWV2WNLgIyBiyn3wizILOTuP6"]

videos: list[YouTube] = [item for sublist in map(lambda x: Playlist(x).videos, playlists) for item in sublist]

print(len(videos))
for idx, yt in enumerate(videos):
    yt.register_on_progress_callback(on_progress)
    st = yt.streams.filter(adaptive=True, file_extension='mp4').order_by('resolution').last()
    print("\nVideo {}/{}".format(idx + 1, len(videos)), '\n', st)
    t = 0
    if st is not None and 'es-419' in yt.captions:
        with open('./raw/' + yt.title.replace('/', '-') + '.vtt', 'w') as subs_file:
            subs_file.write(yt.captions['es-419'].generate_srt_captions())
        st.download('./raw/')
        t += 1
print('{} videos y subtítulos descargados'.format(t))
        