# README #

This repo will contain the code for ICASSP 2019, speaker identifcation (http://www.robots.ox.ac.uk/~vgg/research/speakerID/).

This repo contains a Keras implementation of the paper,     
[Utterance-level Aggregation For Speaker Recognition In The Wild (Xie et al., ICASSP 2019) (Oral)](https://arxiv.org/pdf/1902.10107.pdf).

**New challenge on speaker recognition:
[The VoxCeleb Speaker Recognition Challenge (VoxSRC)](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/competition.html).

### Dependencies
- [Python 2.7-3.6](https://www.continuum.io/downloads)
- [Keras 2.3.1](https://keras.io/)
- [Tensorflow-gpu 1.15.0](https://www.tensorflow.org/)

### Data
The dataset used for the experiments are

- [Voxceleb1, Voxceleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)

### Preprocessing the input files
We have to convert the files to .wav format and convert it to RIFF headers, you can follow these steps

move converttowav.py to dev/aac/ using
- mv converttowav.py dev/aac/

the run this command from the parent folder of the .wav files

- find . -name '*.WAV' | parallel -P20 sox {} '{.}.wav' 


### Training the model
To train the model on the Voxceleb2 dataset, you can run

- python main.py --net resnet34s --batch_size 160 --gpu 2,3 --lr 0.001 --warmup_ratio 0.1 --optimizer adam --epochs 128 --multiprocess 8 --loss softmax --data_path ../path_to_voxceleb2


### Testing the model
To test a specific model on the voxceleb1 dataset, 
for example, the model trained with ResNet34s trained by adam with softmax, and feature dimension 512

- python predict.py --gpu 1 --net resnet34s --ghost_cluster 2 --vlad_cluster 8 --loss softmax --resume ../model/weights.h5


