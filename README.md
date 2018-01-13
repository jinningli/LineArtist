# Souce Code of LineArtist
This is the source code of the paper
**LINE ARTIST: A Multi-style Sketch to Painting Synthesis Scheme**
Arxiv link:
### Requirements
```
Python 3
Pillow (4.2.1)
numpy (1.13.3)
scipy (0.19.0)
opencv-python (3.3.0.10)
tensorflow (1.3.0)
Matlab
```

### Generate Dataset Using Sketch Image Extraction(SIE) Model:
Put your image in the folder ./SIE/SourceImage, then run these command:
```
$ cd ./SIE
$ python3 ./preprocess.py
```
Follow the instructions, dataset will be built in the folder ```./SIE/Datasets```

### Synthesize reality image using Detailed Image Synthesis(DIS) Model:
Please put your dataset inside the folder ```./DIS/Datasets```
#### Train
```
$ cd ./DIS
$ python3 train.py --dataroot ./Datasets/[NAME] --model pix2pix --which_direction AtoB --name [NAME] --gpu_ids 0
```
#### Test
```
$ cd ./DIS
$ python3 test.py --dataroot ./Datasets/[NAME] --model pix2pix --which_direction AtoB --name [NAME] --gpu_ids 0
```
All the checkpoints will be saved in ```./DIS/checkpoints```. The result will be saved in ```./DIS/Results```.

### Stylize using Adaptive Weighted Artist Style Transfer(AWAST) Model:
```
$ cd ./AWAST
$ python3 AWAST.py --content [Path] --folder_styles [Path] --output [Path]
```
### Reference
...
