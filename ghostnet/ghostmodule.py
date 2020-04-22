from math import ceil
import tensorflow as tf
from keras.layers import Conv2D, Concatenate, DepthwiseConv2D, Lambda


class GhostModule(tf.keras.layers.Layer):
    def __init__(self, filters,ratio, convkernel, dwkernel, strides=1):
        super(tf.keras.layers.Layer, self).__init__()
        self.ratio = ratio
        self.filters = filters
        self.conv_out_channel = ceil(filters * 1.0 / ratio)

        self.conv = Conv2D(int(filters), (convkernel, convkernel),
                           strides=(strides,strides), padding='same', activation='relu')
        self.depthconv = DepthwiseConv2D(dwkernel,strides, padding='same',
                                         depth_multiplier=ratio-1, activation='relu')
        self.slice = Lambda(self._return_slices, arguments={'channel': int(self.filters - self.conv_out_channel)})
        self.concat = Concatenate(axis=-1)

    @staticmethod
    def _return_slices(x, channel):
        y = x[:, :, :, :channel]
        return y

    def call(self, x):
        x = self.conv(x)
        if self.ratio == 1:
            return x
        dw = self.depthconv(x)
        dw = self.slice(dw)
        x = self.concat([x, dw])
        return x
