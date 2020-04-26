# ghostnet_tf2
An implementation of GhostNet for Tensorflow 2.1. (From the paper "GhostNet: More Features from Cheap Operations")

Link to paper: https://arxiv.org/pdf/1911.11907.pdf

## Using Ghostnet

This implementation is a normal Keras Model object.
You initialize it and specify the number of classes, build or compile it and it is ready to fit.

Example:
```
from ghost_model import GhostNet

# Initialize model with 1000 classes
model = GhostNet(1000)

# Compile and fit
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy']) 
model.fit(data)

```
