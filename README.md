# FreeSeed: Frequency-band-aware and Self-guided Network for Sparse-view CT Reconstruction
This is the official implementation of the paper "FreeSeed: Frequency-band-aware and Self-guided Network for Sparse-view CT Reconstruction" based on [torch-radon toolbox](https://github.com/matteo-ronchetti/torch-radon/tree/master). The pre-print version can be found in [arxiv](https://arxiv.org/abs/2307.05890); camera-ready version will be soon released.

## Updates
- Aug, 2023: fixed bugs in `dudo_trainer.py`, `basic_wrapper_v2.py` and `dudofree.py`
- Jul, 2023: initial commit.

## Requirements
```
- Linux Platform
- python==3.7.16
- torch==1.7.1+cu110  # depends on the CUDA version of your machine
- torchaudio==0.7.2
- torchvision==0.8.2+cu110
- torch-radon==1.0.0
- monai==1.0.1
- scipy==1.7.3
- einops==0.6.1
- opencv-python==4.7.0.72
- SimpleITK==2.2.1
- numpy==1.21.6
- pandas==1.3.5  # optional
- tensorboard==2.11.2  # optional
- wandb==0.15.2  # optional
- tqdm==4.65.0  # optional
```


## Data Preparation
The AAPM-Myo dataset can be downloaded from: [CT Clinical Innovation Center](https://ctcicblog.mayo.edu/2016-low-dose-ct-grand-challenge/)
(or the [box link](https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h/folder/144594475090)). Please walk through `./datasets/process_aapm.ipynb` for more details on preparing the dataset.



## Training & Inference
Please check `train.sh` for training script (or `test.sh` for inference script) once the data is well prepared. Specify the dataset path and other setting in the script, and simply run it in the terminal.



## Other Notes
We choose torch-radon toolbox because it processes tomography real fast! For those who have problems installing torch-radon toolbox:
- There's other forks of torch-radon like [this](https://github.com/faebstn96/torch-radon) that can be installed via `python setup.py install` without triggering too many compilation errors🤔.
- Check the [issues](https://github.com/matteo-ronchetti/torch-radon/issues) of torch-radon (both open & closed), since there is discussion about any possible errors you may encountered when installing it.



## Citation
If you find our work and code helpful, please kindly cite the corresponding paper:
```
@inproceedings{ma2023freeseed,
  title={FreeSeed: Frequency-band-aware and Self-guided Network for Sparse-view CT Reconstruction},  
  author={Ma, Chenglong and Li, Zilong and Zhang, Yi and Zhang, Junping and Shan, Hongming}, 
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023},
  year={2023}
}
```
