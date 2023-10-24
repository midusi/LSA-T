# LSA-T dataset

Code used to generate LSA-T dataset from public videos found in [CN Sordos](https://www.youtube.com/c/CNSORDOSARGENTINA/playlists) channel.

## Scripts

* ``download.py`` downloads said videos and subtitles into ``raw`` folder.
* ``gen_clips.py`` parses subtitles files (``.vtt``) and generates, for each of the **i** lines of subtitles of the **V** videos:
  * ``data/V/i.mp4`` the clip corresponding to the **i**th line of subtitles.
  * ``data/V/i.json`` that contains:
    * **label**: the line of subtitles corresponding to the clip.
    * **start**: time in seconds where the subtitle starts.
    * **end**: time in seconds where the subtitle ends.
    * **video**: title of the video **V** which the clip belongs to.
  * Parameter ``-d`` (or ``--delete``) deletes both video and subtitle file after processing.
* ``run_ap.sh`` takes as input the path where AlphaPose is installed and runs AlphaPose over all of the generated clips. Output for each **i**th clip is stored in ``data/V/i/alphapose_results.json``
* ``process_ap.py`` infers, in case that there is many people detected by AlphaPose in one clip, which one is the signer. It generates, for each **i**th clip:
  * ``data/V/i_ap.json`` with the raw AlphaPose results for the **i**th video using [Halpe KeyPoints](https://github.com/Fang-Haoshu/Halpe-FullBody) in [AlphaPose default output format](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/output.md).
  * ``data/V/i_signer.json`` that contains:
    * **scores**: for each person in the clip, the amount of "movement" in its hands. It is used to infer who is the signer.
    * **roi**: the considered region of interest of the clip, meaning the box corresponding to the infered signer of the clip.
    * **keypoints**: list of keypoints for each frame of the infered signer in same format that in ``data/V/i_ap.json``.
  * By default it only runs over the videos that haven't been processed yet (there is ``data/V/i/alphapose_results.json`` file, that is deleted after processing and it's content stored in ``data/V/i_ap.json``). Parameter ``-r`` (or ``--rerun``) runs it over already processed files.
* ``gen_extra_dbs`` contains scripts for the generation of extra versions of the database:
  * ``gen_cuts_only_db.py`` generate a database in ``data/cuts_only`` that contains only the videos and metadata (excluding keypoint info)
  * ``gen_vis_db.py`` generates a lightweight version of the database with videos in lower quality and that have the keipoints and roi embebbed on them. It is used for visualization.


## Visualization

Stored in ``visualization`` show visualizations and statistics about the database content. They do not take part in the database generation and processing.

* ``subtitles_analysis.ipynb`` shows statistics about the clips duration, words per clip, most used words and n-grams.
* ``signers_analysis.ipynb`` shows statistics about amount of people per clip and confidence about the one chosed as signer.
* ``fiftyone_visualization.py`` starts a [FiftyOne](https://voxel51.com/docs/fiftyone/) sessions and loads the database for visualization. First time might take a while to build the dataset in the correct format.
  * Parameter ``--full`` or ``-f`` is used to load the ful version of the dataset (with HQ videos and keypoint and roi live data). This may need a lot of memory.
