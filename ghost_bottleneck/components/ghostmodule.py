from math import ceil
from tensorflow.keras.layers import Conv2D, Concatenate, DepthwiseConv2D, Lambda, Layer, Activation


class GhostModule(Layer):
    def __init__(self, filters, ratio, convkernel, dwkernel, use_bias, strides=1):
        super(Layer, self).__init__()
        self.ratio = ratio
        self.filters = filters
        self.conv_out_channel = ceil(filters * 1.0 / ratio)

        self.conv = Conv2D(int(filters), (convkernel, convkernel), use_bias=use_bias,
                           strides=(strides,strides), padding='same', activation=None)
        self.depthconv = DepthwiseConv2D(dwkernel, strides, padding='same', use_bias=use_bias,
                                         depth_multiplier=ratio-1, activation=None)
        self.relu = Activation('relu')
        self.slice = Lambda(self._return_slices, arguments={'channel': int(self.filters - self.conv_out_channel)})
        self.concat = Concatenate(axis=-1)

    @staticmethod
    def _return_slices(x, channel):
        y = x[:, :, :, :channel]
        return y

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.relu(x)
        if self.ratio == 1:
            return x
        dw = self.depthconv(x)
        dw = self.relu(dw)
        dw = self.slice(dw)
        x = self.concat([x, dw])
        return x
