from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Layer, BatchNormalization, Activation, add

from ghost_bottleneck.components.semodule import SEModule
from ghost_bottleneck.components.ghostmodule import GhostModule


class GBNeck(Layer):
    def __init__(self, dwkernel, strides, exp, out, ratio, use_se):
        super(GBNeck, self).__init__()
        self.strides = strides
        self.use_se = use_se
        self.batchnorm = BatchNormalization()
        self.conv = Conv2D(out, (1, 1), strides=(1, 1), padding='same', data_format='channels_last',
                           activation=None, use_bias=False)
        self.relu = Activation('relu')
        self.depthconv = DepthwiseConv2D(dwkernel, strides, padding='same', depth_multiplier=ratio-1,
                                         data_format='channels_last', activation=None, use_bias=False)
        self.ghost1 = GhostModule(exp, ratio, 1, 3, use_bias=False)
        self.ghost2 = GhostModule(out, ratio, 1, 3, use_bias=False)
        self.se = SEModule(exp, ratio)

    def call(self, inputs):
        x = self.depthconv(inputs)
        x = self.batchnorm(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        y = self.ghost1(x)
        y = self.batchnorm(y)
        y = self.relu(y)
        # Extra depth conv if strides higher than 1
        if self.strides > 1:
            y = self.depthconv(y)
            y = self.batchnorm(y)
            y = self.relu(y)
        # Squeeze and excite
        if self.use_se:
            y = self.se(y)
        y = self.ghost2(y)
        y = self.batchnorm(y)
        # Skip connection
        return add([x, y])
