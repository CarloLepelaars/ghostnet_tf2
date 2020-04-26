import tensorflow as tf
from ghost_model import GhostNet

if __name__ == "__main__":
    # Initialize model with 1000 classes
    model = GhostNet(1000)

    # Build model with specified shape
    model.build((None, 224, 224, 3))

    # Compile and fit
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam', metrics=['accuracy'])
    # model.fit(tf.ones((8, 224, 224, 3)))
    print(model.summary())
