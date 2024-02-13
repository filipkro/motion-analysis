The structure of this project is a bit convoluted, sorry about that..

This repo can be seen as the top level, mainly containing code to generate datasets. Running install-repos script as bellow will clone repos for keypoint extraction in videos ([mmpose](https://github.com/filipkro/mmpose.git) and [mmdet](https://github.com/filipkro/mmdetection.git)) into `pose/` as well as code to train and validate models for classification of the motions (from https://github.com/filipkro/tsc.git).

Both keypoint extraction and model training can probably be run locally, but will take some time unless you have some compute (preferably with GPUs)... During my MSc thesis I had access to the [Alvis cluster](https://www.c3se.chalmers.se/about/Alvis/).

## Overview
For running the keypoint extraction [run-detection](run-detection) adn [run-detection-folder](run-detection-folder) are example sbatch files running the extraction on a single video or a folder on Alvis (note, they worked 3 years ago, I don't know what has changed now etc, I didn't use containers for this unfortunately...).

Running on a folder will call [`pose/analysis/run-folder-cluster.sh`](pose/analysis/run-folder-cluster.sh) which calls [`analyse_folder.py`](pose/analysis/analyse_folder.py) in the same directory. This will in turn run keypoint extraction ([`analyse_vid.py`](pose/analysis/analyse_vid.py)) on all videos in the specified directory (how this path is spoecified and resolved might have to be changed depending on where and how it is run and organised). When extracting keypoints the frames are flipped such that all exercises are "conducted" on the right leg. Some videos had this information in the file name, others hadn't. When it couldn't be identified from the filename the keypoint positions were analysed - how this is done needs top be modified for other motions.

In `pose/analysis/utils` there are a bunch of `create_POE_*.py` files. These are used to create the actual dataset used for specific POEs. For the different POEs different keypoints are used etc. These are atm incredibly messy - I will write a new one generating test and train sets in a more comprehensible way... Note that for such train and test sets repetitions from the same individual should not be in both datasets. And finding which keypoints to use for different POEs wasn't trivial, a lot of time can be spent on this. I looked at gradients for different channels to try to identify important features (more info on this in the classification code, and I wrote a bit about in the thesis).

I will rewrite the `create_POE` script to make it clearer, but haven't had time yet...

# motion-analysis

master's thesis project, assessments of POEs in videos.
report: [Visual assessments of Postural Orientation Errors using ensembles of Deep Neural Networks](https://github.com/filipkro/motion-analysis/blob/master/tex/mt-motion-analysis.pdf)

## install repos:
```
$ cd motion-analysis
$ ./scripts/install-repos.sh
```

I ran with Python Python 3.7

install dependencies for mmpose (preferably from within some virtual env):
```
$ ./install/install.sh
```
