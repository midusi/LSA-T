## LSA-T: The first continuous LSA dataset

LSA-T is the first continuous Argentinian Sign Language (LSA) dataset. It contains 14,880 sentence level videos of LSA extracted from the [CN Sordos YouTube channel](https://www.youtube.com/c/CNSORDOSARGENTINA) with labels and keypoints annotations for each signer. Videos are in 30 FPS full HD (1920x1080).

* [Download link](http://c1781468.ferozo.com/data/lsa-t.7z) (45GB compressed)
* [Visualization notebook](https://colab.research.google.com/drive/1kj5ztYw_57fi6wo2dpL18UkBR9ciV6ki)
* Presentation paper (**TO-DO**)

|                                               |                                               |                                               |
|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|
| <img width="100%" src="docs/assets/clip2.gif"> | <img width="100%" src="docs/assets/clip3.gif"> | <img width="100%" src="docs/assets/clip1.gif"> |

### Format

Samples are organized in directories according to the playlists and video they belong to. For each sample ```i``` there are four files:

* ``i.mp4``: the clip corresponding to the ith line of subtitles.
* ``i.json`` contains:
    * **label**: the line of subtitles corresponding to the clip.
    * **start**: time in seconds where the subtitle starts.
    * **end**: time in seconds where the subtitle ends.
    * **video**: title of the video which the clip belongs to.
    * **playlist**: title of the playlist which the clip belongs to.
* ``i_ap.json``: the raw AlphaPose results over the clip using [Halpe KeyPoints](https://github.com/Fang-Haoshu/Halpe-FullBody) in [AlphaPose default output format](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/output.md).
* ``i_signer.json`` contains:
    * **scores**: for each person in the clip, the amount of "movement" in its hands. It is used to infer who is the signer.
    * **roi**: the considered region of interest of the clip (bounding box of the infered signer).
    * **keypoints**: list of keypoints for each frame of the infered signer in same format that in ``i_ap.json``.

### Usage

This repository can be installed via ``pip`` and contains the [``LSA_Dataset``](https://github.com/midusi/LSA-T/blob/main/lsat/dataset/LSA_Dataset.py) class (in ``lsat.dataset.LSA_Dataset`` module). This class inherits from the [Pytorch dataset class](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) and implements all necessary methods for using it with a Pytorch dataloader. It also manages the downloading and extraction of the database.

Also, useful transforms for the clips and keypoints are provided in [``lsat.dataset.transforms``](https://github.com/midusi/LSA-T/blob/main/lsat/dataset/transforms.py)

### Statistics and comparison with other DBs

|                     | **LSA-T**          | **PHOENIX***     | **SIGNUM**      | **CSL**            | **GSL**     | **KETI**           |
|-------------------------|--------------------|------------------|-----------------|--------------------|-------------|--------------------|
| **language**            | Spanish            | German           | German          | Chinese            | Greek       | Korean             |
| **sign language**       | LSA                | GSL              | GSL             | CSL                | GSL         | KLS                |
| **real life**           | **Yes**            | **Yes**          | No              | No                 | No          | No                 |
| **signers**             | **103**            | 9                | 25              | 50                 | 7           | 14                 |
| **duration (h)**        | 21.78              | 10.71            | 55.3            | **100+**           | 9.51        | 28                 |
| **# samples**           | 14,880             | 7096             | **33,210**      | 25,000             | 10,295      | 14,672             |
| **# unique sentences**  | **14,254**         | 5672             | 780             | 100                | 331         | 105                |
| **% unique sentences**  | **95.79%**         | 79.93%           | 2.35%           | 0.4%               | 3.21%       | 0.71%              |
| **vocab. size (w)**     | **14,239**         | 2887             | N/A             | 178                | N/A         | 419                |
| **# singletons (w)**    | **7150**           | 1077             | 0               | 0                  | 0           | 0                  |
| **% singletons (w)**    | **50.21%**         | 37.3%            | 0%              | 0%                 | 0%          | 0%                 |
| **vocab. size (gl)**    | -                  | **1066**         | 450             | -                  | 310         | 524                |
| **# singletons (gl)**   | -                  | **337**          | 0               | -                  | 0           | 0                  |
| **# singletons (gl)**   | -                  | **31.61%**       | 0%              | -                  | 0%          | 0%                 |
| **resolution**          | **1920x1080**      | 210x260          | 776x578         | **1920x1080**      | 848x480     | **1920x1080**      |
| **fps**                 | **30**             | 25               | **30**          | **30**             | **30**      | **30**             |

\*Data was not available for the whole PHOENIX dataset, so the table show its train set statistics.

### Evaluation splits

<table>
    <tr>
        <td></td>
        <td><b>LSA-T</td>
        <td colspan=2><b>Full version</td>
        <td colspan=2><b>Reduced version</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td>Train</td>
        <td>Test</td>
        <td>Train</td>
        <td>Test</td>
    </tr>
    <tr>
        <td><b>signers</td>
        <td>103</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
    </tr>
    <tr>
        <td><b>duration [h]</td>
        <td>21.78</td>
        <td>17.49</td>
        <td>4.29</td>
        <td>15.85</td>
        <td>3.89</td>
    </tr>
    <tr>
        <td><b># sentences</td>
        <td>14,880</td>
        <td>11,065</td>
        <td>2735</td>
        <td>3767</td>
        <td>910</td>
    </tr>
    <tr>
        <td><b>% unique sentences</td>
        <td>95.79%</td>
        <td>96.64%</td>
        <td>92.78%</td>
        <td>96.88%</td>
        <td>98.35%</td>
    </tr>
    <tr>
        <td><b>vocab. size</td>
        <td>14,239</td>
        <td>12,385</td>
        <td>5546</td>
        <td>2694</td>
        <td>1579</td>
    </tr>
    <tr>
        <td><b>% singletons</td>
        <td>50.21%</td>
        <td>52.01%</td>
        <td>61.9%</td>
        <td>23.2%</td>
        <td>48.83%</td>
    </tr>
    <tr>
        <td><b>% sentences with singletons</td>
        <td>34.97%</td>
        <td>40.98%</td>
        <td>67.97%</td>
        <td>14.36%</td>
        <td>54.29%</td>
    </tr>
    <tr>
        <td><b>% sentences with words not in train vocabulary</td>
        <td>-</td>
        <td>-</td>
        <td>59.2%</td>
        <td>-</td>
        <td>84.5%</td>
    </tr>
</table>

### Citation

```
@article{bianco2022lsa,
  title={LSA-T: The first continuous Argentinian Sign Language dataset for Sign Language Translation}, 
  author={Bianco, Pedro Dal and R{\'\i}os, Gast{\'o}n and Ronchetti, Franco and Quiroga, Facundo and Stanchi, Oscar and Hasperu{\'e}, Waldo and Rosete, Alejandro},
  journal={arXiv preprint arXiv:2211.15481},
  year={2022}
}
```
