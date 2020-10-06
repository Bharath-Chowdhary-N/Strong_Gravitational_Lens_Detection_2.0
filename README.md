# Strong Gravitational Lens Detection 2.0
This project is about detecting strong gravitational lenses in the Kilo Degree Survey.

## Based on
This project is based on the following project: https://github.com/CEnricoP/cnn_strong_lensing & paper: https://arxiv.org/abs/1702.07675

## Requirements
Use the following command to install required packages: 
~~~
pip3 install -r requirements.txt
~~~
If you wish to make use of the max-tree segmentation as preprocessing step, then you need to take the following steps:
- Get the zip file "siamxt-master.zip" from: https://github.com/rmsouza01/siamxt
- Transfer this file to the machine that you will use (for example Peregrine)
- Load Python, for example I would do this on Peregrine: (This forces Python to be loaded)
	- module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4
	- module load matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
	- module load scikit-image/0.16.2-fosscuda-2019b-Python-3.7.4
- Now that python is loaded you can install using: (This installs directly from .zip file)
~~~
pip3 install siamxt-master.zip --user.
~~~
If succesful, then the parameter "do_max_tree_seg" can be set to True.



## How to run
~~~
python3 main.py --run=runs/experiment_folder/run.yaml
~~~

Input parameters of a run can be set in the following file: runs/experiment_folder/run.yaml.

## Data
Data can be requested.

The data is split in the following way:
- Train Data      80%
- Validation Data 10%
- Test Data       10%

In this binary classification problem three types of images with dimensions (101,101,1) are used:
- 100000 Sources  (Simulated Lensing features as *.fits* files.)
- 5513 Lenses     (An image of a galaxy probalby not showing strong gravitational lensing features.)
- 6083 Negatives  (An image identified as not showing strong gravitational lensing features.)


More detailed information will be added later on. At this stage in the project, changes will be frequent.

