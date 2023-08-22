# Non-uniform Speaker Disentanglement for Depression Detection From Raw Speech Signals

[Screenshot](NUSD%20IS2023%20Poster.pdf)

## **Credit**

This repository relates to our work in the Interspeech 2023 paper, "Non-uniform Speaker Disentanglement For Depression Detection From Raw Speech Signals". 

Camera-ready preprint - https://arxiv.org/abs/2306.01861

The framework is based on https://github.com/adbailey1/DepAudioNet_reproduction

**Prerequisites**

This was develpoed for Ubuntu (18.04) with Python 3

Install miniconda and load the environment file from environment.yml file

`conda env create -f environment.yml`

(Or should you prefer, use the text file)

Activate the new environment: `conda activate myenv`

## **Dataset**

For this experiment, the DAIC-WOZ dataset is used (found in AVEC16 - AVEC17, 
not the newer extended version). This can be 
obtained
 through The University of Southern California (http://dcapswoz.ict.usc.edu
 /) by signing an agreement form. The dataset is roughly 135GB. 
 
 The dataset contains many errors and noise (such as interruptions during an
  interview or missing transcript files for the virtual agent). It is
   recommended to download and run my DAIC-WOZ Pre-Processing Framework in
    order to quickly begin experimenting with this data (https://github.com/adbailey1/daic_woz_process)

## **Experiment Setup**

Use the config file to set experiment preferences and locations of the code, workspace, and dataset directories. Configuration controlling experiment can be modified in `config_disent_raw_grad.py`. Description is provided inside the config script.

Optional commands are: 
- `--validate` - to train a model with a validation set
- `--cuda` - to train a model using a GPU
  
TEST MODE ONLY:
-  `--prediction_metric` - this determines how the output is calculated 
   running on the test set in test mode. 0 = best performing model, 1 = 
   average of all 
   models, 2 = majority vote from all models  
- `--threshold` - this determines the selection of the best model in each `EXP_RUNTHROUGH`. total = best model considering train/dev loss/F1-avg, fscore = best model with best validation F1-avg. 
- Reported results can be obtained by setting `--prediction_metric=1 --threshold=total`

For example: To run a training experiment without bash, using a validation
 set, GPU.
 
 `python3 main_disent_fscore_grad.py train --validate --cuda --position=1`
 
 Running trained models again on the validation set:
 
 `python3 main_disent_fscore_grad.py test --validate --cuda --prediction_metric=1 --threshold=total`
 
## Pre-trained Models
The best models without disentanglement, Uniform Speaker Disentanglement (USD) and Non-uniform Speaker Disentanglement (NUSD) are shared. Please download from the Google Drive https://drive.google.com/file/d/1ILgYpktXEZq2f_1IefBy_3UYy7TdWxFg/view?usp=share_link

Once downloaded, change `EXPERIMENT_DETAILS['SUB_DIR']` into the experiment directory and run test script.


## Citations

If you find this repo useful in your work, please cite the following  - 

`@misc{wang2023nonuniform,
      title={Non-uniform Speaker Disentanglement For Depression Detection From Raw Speech Signals}, 
      author={Jinhan Wang and Vijay Ravi and Abeer Alwan},
      year={2023},
      eprint={2306.01861},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}`

`@inproceedings{ravi22_interspeech,
  author={Vijay Ravi and Jinhan Wang and Jonathan Flint and Abeer Alwan},
  title={{A Step Towards Preserving Speakers’ Identity While Detecting Depression Via Speaker Disentanglement}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={3338--3342},
  doi={10.21437/Interspeech.2022-10798}
}`




