import os, unicodedata, re
from pytube import Playlist, YouTube
from pytube.cli import on_progress

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

def main():
    # Playlists de resumen semanal, ecologia, ultimo momento y #leyfederalLSA
    playlists = ["https://www.youtube.com/playlist?list=PLhysX0rYXWV2xM3T4KAqaAEg-GEhDXwai",
                "https://www.youtube.com/playlist?list=PLhysX0rYXWV2DioiHvMjzJs_4UWifmPo9",
                "https://www.youtube.com/playlist?list=PLhysX0rYXWV2wQnK2nxU4gvcmq5yrHli3",
                "https://www.youtube.com/playlist?list=PLhysX0rYXWV2WNLgIyBiyn3wizILOTuP6"]

    videos: list[YouTube] = [item for sublist in map(lambda x: Playlist(x).videos, playlists) for item in sublist if 'es-419' in item.captions]

    for idx, yt in enumerate(videos):
        yt.register_on_progress_callback(on_progress)
        st = yt.streams.filter(adaptive=True, file_extension='mp4').order_by('resolution').last()
        print(f"\nVideo {idx + 1}/{len(videos)}: {yt.title.replace('/', '-')}\n{st}")
        if not os.path.isfile('./raw/' + slugify(yt.title) + '.vtt') and st is not None:
            st.download(output_path='./raw/', filename=slugify(yt.title) + '.mp4')
            with open('./raw/' + slugify(yt.title) + '.vtt', 'w') as subs_file:
                subs_file.write(yt.captions['es-419'].generate_srt_captions())

if __name__ == "__main__":
    main()
