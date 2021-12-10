# CSCI1470FinalProj

## Directory structure

**eval_sample**: A sample input and label image that can be used to help debug the model.

**model**: Contains code for loading image data (*get_data.py*) and running the model (*model.py*). Usage is *python model.py*. Note that model.py expects data to be in a *data* folder at the top level; we could not upload our data to github due to insufficient space. Please email nicholas_petrocelli@brown.edu for google drive access if you wish to download it.

**outputs**: Folder for holding sampled model output.

**test_run_records**: folders that contain records of various implementation tests that we ran. Each contains at least one .csv file (pipe-separated) with record of training and testing loss/accuracy over several epochs, as well as sampled output from the model. The *arch_test* subfolder represent architecture tests; see the *model_arch_x.txt* files to see what architecture was being tested in each run.

**tif_processing.py**: code used to preprocess and slice the raw .tif images into the 100x100 .png images we used for training/testing.

**requirements.txt**: Lists some packages not included in the commonly used homework packages that are necessary for this project. 