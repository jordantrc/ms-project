# MS Project

Helper scripts for the MS project.

## Datasets

*  HMDB
*  UCF101
*  Sports-1m

## Workflow:

1.  Sample all videos in the dataset, sampling method is to randomly sample five 2-second clips from the video
2.  Generate TF records from the sampled video frames
3.  Split the tfrecords into training and validation sets (uniform distribution across classes)
4.  Create Interval Algebra Descriptor (IAD) images from the input data
5.  Feed the IAD images into the 2DCNN for training
6.  Use the validation set to test the accuracy of the IAD 2DCNN

