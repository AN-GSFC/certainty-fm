# certainty-fm
This repository implements MC dropout uncertainty estimation for the Prithvi-100M foundation model.

There are also conformal predictions and calibration evaluations. 

## Setup

Please follow the setup instructions in this repository:
https://github.com/NASA-IMPACT/hls-foundation-os

Alternatively, create the conda environment, activate it, then run setup.sh

Then, download the Sen1Floods11 dataset 

```
gsutil -m rsync -r gs://sen1floods11 .
```

There are some labeled examples already provided in the test_labels directory.

To download more, you can use 

```
curl -O https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/{image_name}
```

The input images associated with these labels can be found in /data/flood_events/Hand_Labeled/S2Hand. Please put these images in a directory called test_images.

The set of images the model performs MC dropout on is dependent on how many images are provided in the test_images directory. 

## Usage

prithvi_mcdropout.py
```
ARGS: 
--gpu (number of gpus used. specify 0 for no gpus. multi-gpu training currently not working.)
--stop (int) (include the flag with an integer n, to stop inference after image n. for inference of all images in /test_images, specify this as -1)
--mc (int) (specify number of montecarlo dropout trials for certainty estimation. default is 3.)

e.g.
python prithvi_mcdropout.py --gpu 1 --stop 5 --mc 5
```
