# WiderFace-Evaluation
Python Evaluation Code for [Wider Face Dataset](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)


## Usage


##### Setup:

````
pip install Cython
python3 setup.py build_ext --inplace
````

##### Evaluation

**Ground Truth dir should contain:** `wider_face_val.mat`, `wider_easy_val.mat`, `wider_medium_val.mat`,`wider_hard_val.mat`

````
python3 evaluation.py -p <your prediction dir> -g ground_truth
````

## Acknowledgements

Code borrowed from https://github.com/wondervictor/WiderFace-Evaluation
