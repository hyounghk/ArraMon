# ArraMon / ऐरेमौन
Code/dataset/simulator for EMNLP 2020 findings paper ["ArraMon: A Joint Navigation-Assembly Instruction Interpretation Task in Dynamic Environments"](http://arxiv.org/abs/2011.07660) Hyounghun Kim,  Abhay Zala, Graham Burri, Hao Tan, Mohit Bansal. 
<p align="center">
  <a href="https://arramonunc.github.io">arramonunc.github.io</a>
</p>
<p align="center">
  <img src="https://github.com/hyounghk/ArraMon/blob/main/final-english-hindi.gif">
</p>

### \*Hindi dataset is being added (val-seen and val-unseen splits are uploaded in [data](./data) folder).


## Prerequisites

- Python 3.6
- [PyTorch 1.4](http://pytorch.org/) or Up


### Simulator Setup:
The simulator will come soon.


## Usage

To train the model:
```
python src/main.py --port PORT --batch_size BATCH_SIZE --sim_num SIM_NUM
```
PORT: port number (should be matched to the port number of the simulator). <br>
BATCH_SIZE: batch size. <br>
SIM_NUM: the number of simulators to run. <br>

Then, run the simulator (note that the model should be started first, then the simulator next).
#### Sim on CPU 
```
cd simulator
sh run.sh SIM_NUM SIM_BATCH PORT
```
SIM_BATCH: batch size of each sim. Equal to BATCH_SIZE/SIM_NUM.

#### Sim on GPU
```
cd PATH_TO_THE_SIM/build
```
put run_gpu_sim.sh in PATH_TO_THE_SIM/build.
```
sh run_gpu_sim.sh SIM_NUM SIM_BATCH PORT
```
\* Note: Currently, sim-gpu only supports SIM_NUM=1.<br>
### Pre-Recorded Image Features
If you are using teacher-forcing training and want to reduce training time, please consider using pre-recorded image features.
Please download the image features from [here](https://drive.google.com/file/d/1-Wchv0sK4B4L9Lcof1_34q64zIjNS2mC/view?usp=sharing) and the position data from [here](https://drive.google.com/file/d/1j7g4qNBIK-9bG6Q4nF2BPlJBu1C1K8ej/view?usp=sharing) and unzip in the root folder.

Use this command to run the model:
```
python src/main_nosim.py --port PORT --batch_size BATCH_SIZE --batch_size_val BATCH_SIZE_VAL --sim_num SIM_NUM
```
BATCH_SIZE_VAL: batch size of validation dataset split.

This allows you to train your model with pre-recorded features and evaluate it with data from the simulator.
### Evaluation on Test split
Please contact arramonunc@gmail.com for the test split.

### Citation
```
@inproceedings{Kim2020ArraMonAJ,
  title={ArraMon: A Joint Navigation-Assembly Instruction Interpretation Task in Dynamic Environments},
  author={Hyounghun Kim and Abhaysinh Zala and Graham Burri and Hao Tan and Mohit Bansal},
  booktitle={Findings of EMNLP},
  year={2020}
}
```

## Acknowledgments
The nDTW calculation code is borrowed from the [R4R code repository](https://github.com/google-research/google-research/tree/master/r4r). <br>
Some code is from ["Expressing Visual Relationships via Language" paper's code repository](https://github.com/airsplay/VisualRelationships).


## Disclaimers
We use [mapbox](https://www.mapbox.com/) for our map data. 
It it required to add the mapbox logo when you publish any image from the map. So please use the image file below to place the logo in images that you publish for visualiztion. <br>
![maobox Logo](https://github.com/hyounghk/ArraMon/blob/main/mapbox-attribution.png)
