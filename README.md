# ConvCRFs-training
The code is a realization of training code of ConvCRFs

The file convcrf.py is modified from [https://github.com/MarvinTeichmann/ConvCRF](https://github.com/MarvinTeichmann/ConvCRF), and part of training codes are modified from the kaggle notebook [https://www.kaggle.com/code/jokerak/fcn-voc](https://www.kaggle.com/code/jokerak/fcn-voc)

## Realization

I tried to train unary (fcn-resent101) using the same config as the paper, and here is the result

|  Unary   | epochs |  Global ACC |  mIoU   |
| --- | -------- | -------- | --- |
| paper    |  200 |    91.84      |  71.23   |
| ours    | 25+ft | 93.33    |  71.00    |

Besides, I also tried to add some feature vectors into message passing function (+C means choose conv1x1 as the compatibility transformation and +F means add unary output as the feature vector), the origin paper say that they using 11x11 filter size, and we are using 7x7 filter size

|  CRFs   | method |  Global ACC |  mIoU   |
| --- | -------- | -------- | --- |
| paper    | +C (11) |    94.01      |  72.30   |
| ours    | +C (7) | 93.51    |  71.85    |
| ours    | +C + F | 93.52    |  71.92    |

### Folder structure

To train your own SBD dataset or Pascal VOC dataset, the folder structure are forced to be

VOC2012/                                                                                                       
--Annotations/                                                                                                       
--ImageSets/                                                                                                       
--JPEGImages/                                                                                                       
--SegmentationClass/                                                                                                       
--SegmentationClassAug/  
----trainaug.txt                                                                                                       
----val.txt                                                                                                       
  
and make sure to add trainaug.txt to your "ImageSets/Segmentation/trainaug.txt"

### Custom Dataset

If your are interesting in training your own dataset, just to modify the dataset.py file and the training configuration. 

### Disscussion

- We found that the more feature vectors we added or increased the filter size, the more mIoU dropped and didn't recover
- The mIoU value of jointly end-to-end training is less than freeze unary parameters.

### Usage

```conda
python main.py
```
