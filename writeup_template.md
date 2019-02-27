## Project: Deep Learning

### Introduction

This report provides detail of the implementation for "follow me" project which trains a deep neural network to identify and track a target in simulation. 

#### Network architecture

So far, the fully connected layers, which is good at tasks of image classification, can only tell what/who is inside the frame. This project applies fully convolutional network to tell the location of the object  in the frame. There are 2 stages of the network: (1) encoder, and (2) decoder. Those 2 stages are connected by 1x1 convolutional layer.

#### Training and Experimenting

I ended up to use these hyper-parameters. It gives the final IOU about 0.44 with the run-time on GPU about 1 hour.

learning_rate = 0.005
batch_size = 32
num_epochs = 100
steps_per_epoch = 100
validation_steps = 50
workers = 4

The picture below is the final loss after 100 epochs.
![final loss](img/final_loss.png)

#### Results and Conclusion

I did not have a chance to capture extra data. Testing with the sample data produce the expected ~0.4 IOU. I think the model is good to identify and track the object.