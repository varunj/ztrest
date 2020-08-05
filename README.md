# Obtaining High-Quality Expressions Label from ArgentLite 

### Varun Jain / Summer 2020



## Problem Statement
Given 11-view Rosetta results on particular ALPP capture, we trained vanilla HMC face encoders which don’t generalize well to real-world variations such as lighting, appearance (e.g. beard), and headset assembly (i.e. relative camera positions). Methods including augmentation and normalization in the input have been pursued, but these methods are not aware of available information during test time: target image (with lighting/appearance information) and target camera parameters (with headset assembly information). Therefore, we can only blindly expand augmentation that leads to degraded precision when the model capacity runs out. On the other hand, training 3-view Rosetta can utilize those available information, but the results are sub-optimal without utilizing knowledge transferred from 11-view data. In this project, we aim at obtaining high quality labels based on a 3-view Rosetta setting, with exploiting a large number of 11-view Rosetta pairs on multi-identity data. The idea is to learn a generic feature space which both input IR images and mugsy texture are transformed towards through fully convolutional networks (so pixels are aligned), and comparison in this feature space minimizes the distance between current code/slop to 11-view results.

The methods explored were mostly centered around metric learning and mimicking the 11-view Rosetta landscape.



## Setup Logistics

### Documentation
- All the experiment details can be found at
https://docs.google.com/document/d/1k1-ejVlVOvDYlC7vfH3ovuQXMBfNhU4CxifxLu4Q5T4/
- The presentation slides explaining the results can be found at
https://docs.google.com/presentation/d/1udC95saSGZoJHc22LaUlHk-bHEa9b8tdZ5_o4kdbwzI/edit#slide=id.g8586b6d298_0_3/


### Folder Structure
 - All the experiments use a common data loader that can be found at
`data_loader/data_handler_multi_identity_3view.py`
 
 - All the code can be found in the following directory
`ls multi_identity_3view/`

 - The associated config files are
`ls configs/multi_identity*.json`

 - Scripts to run the experiments are
`ls scripts/run_multi_identity_*.py`

 - Result output directories can be found at
`ls /mnt/home/varunj/outs*`
each folder has `log`, `model`, `video` folders that contain the train/inference Tensorboard stats, saved models, and, train/inference videos respectively. 


### Virtual Environment
`source /mnt/home/gbschwartz/anaconda/bin/activate dgxenv`


### Compile Rendering Layer
Suitably change lib paths in
`render_new/drtk/src/modules/rasterize_module.cpp`
```
cd render_new/drtk
python setup.py build_ext --inplace
```



## How to Run

### Baselines
- Baseline 0
`python scripts/run_multi_identity_3view_baseline0.py configs/multi_identity_baseline0_*.json --no_save`

- Baseline 4
`python scripts/run_multi_identity_3view_baseline4.py configs/multi_identity_baseline4.json --no_save`

- Baseline 1 and 3 - Training \
`python scripts/run_multi_identity_3view.py configs/multi_identity_baseline1_*.json --train`
`python scripts/run_multi_identity_3view.py configs/multi_identity_baseline3_*.json --train`

- Baseline 1 and 3 - Inference
`python scripts/run_multi_identity_3view.py configs/multi_identity_baseline1_*.json --no_save`
`python scripts/run_multi_identity_3view.py configs/multi_identity_baseline3_*.json --no_save`


### Metric Learning
- Train a model
`python scripts/run_multi_identity_3view_metric_learn.py configs/multi_identity_metric.json --train`

- Generate visualizations of outputs of the models
`python scripts/run_multi_identity_3view_metric_learn.py configs/multi_identity_metric.json --no_save` 

- Inference
`python scripts/run_multi_identity_3view_metric_run.py configs/multi_identity_metric.json --no_save` 


### Mimicking 11-view Rosetta Landscape
- Generate PCA results
`python scripts/run_multi_identity_3view_mimic11v_run.py configs/multi_identity_mimic11v.json --train`

- Train a model
`python scripts/run_multi_identity_3view_mimic11v_learn.py configs/multi_identity_mimic11v.json --train`

- Generate visualizations of outputs of the models
`python scripts/run_multi_identity_3view_mimic11v_learn.py configs/multi_identity_mimic11v.json --no_save` 

- Inference
`python scripts/run_multi_identity_3view_mimic11v_run.py configs/multi_identity_mimic11v.json --no_save` 
