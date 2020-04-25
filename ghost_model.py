from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D, Lambda

from ghost_bottleneck.bottleneck import GBNeck


class GhostNet(Model):
    """
    The main GhostNet architecture as specified in "GhostNet: More Features from Cheap Operations"
    Paper:
    https://arxiv.org/pdf/1911.11907.pdf
    """
    def __init__(self, classes):
        super(GhostNet, self).__init__()
        self.classes = classes
        self.conv1 = Conv2D(16, (3, 3), strides=(2, 2), padding='same',
                            activation=None, use_bias=False)
        self.conv2 = Conv2D(960, (1, 1), strides=(1, 1), padding='same', data_format='channels_last',
                            activation=None, use_bias=False)
        self.conv3 = Conv2D(1280, (1, 1), strides=(1, 1), padding='same',
                            activation=None, use_bias=False)
        self.conv4 = Conv2D(self.classes, (1, 1), strides=(1, 1), padding='same',
                            activation=None, use_bias=False)
        self.batchnorm = BatchNormalization(axis=-1)
        self.relu = Activation('relu')
        self.softmax = Activation('softmax')
        self.squeeze = Lambda(self._squeeze)
        self.pooling = AveragePooling2D(pool_size=(7, 7))

        self.dwkernels = [3, 3, 3, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5]
        self.strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]
        self.exps = [16, 48, 72, 72, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960, 960, 960]
        self.outs = [16, 24, 24, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 160, 160]
        self.ratios = [2] * 16
        self.use_ses = [False, False, False, True, True, False, False, False,
                        False, True, True, True, False, True, False, True]
        for i, args in enumerate(zip(self.dwkernels, self.strides, self.exps, self.outs, self.ratios, self.use_ses)):
            setattr(self, f"gbneck{i}", GBNeck(*args))

    @staticmethod
    def _squeeze(x):
        return K.squeeze(x)

    def call(self, inputs):
        x = self.relu(self.batchnorm(self.conv1(inputs)))
        for i in range(16):
            x = getattr(self, f"gbneck{i}")(x)
        x = self.relu(self.batchnorm(self.conv2(x)))
        x = self.pooling(x)
        x = self.relu(self.batchnorm(self.conv3(x)))
        x = self.conv4(x)
        x = self.squeeze(x)
        output = self.softmax(x)
        return output


if __name__ == "__main__":
    model = GhostNet(1000)
    model.build(input_shape=(None, 224, 224, 3))
    print(model.summary())
