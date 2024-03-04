# Dataset Preparation

We provide our labels in `data_list`.

## InfAR



1. Download the videos.

2. After all the videos were downloaded, resize the video to the short edge size of 224/256, then prepare the csv files for training, validation, and testing set as `train.csv`, `val.csv`, `test.csv` in `data_list/infar`. The format of the csv file is:

```
path_to_video_1,label_1
path_to_video_2,label_2
path_to_video_3,label_3
...
path_to_video_N,label_N
```


