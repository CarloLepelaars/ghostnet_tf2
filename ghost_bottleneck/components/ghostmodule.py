from math import ceil

from tensorflow.keras.layers import Conv2D, Concatenate, DepthwiseConv2D, Lambda, Layer, Activation


class GhostModule(Layer):
    """
    The main Ghost module
    """
    def __init__(self, out, ratio, convkernel, dwkernel, use_bias, strides=1):
        super(GhostModule, self).__init__()
        self.ratio = ratio
        self.out = out
        self.conv_out_channel = ceil(self.out * 1.0 / ratio)

        self.conv = Conv2D(self.out, (convkernel, convkernel), use_bias=use_bias,
                           strides=(strides, strides), padding='same', activation=None)
        self.depthconv = DepthwiseConv2D(dwkernel, strides, padding='same', use_bias=use_bias,
                                         depth_multiplier=ratio-1, activation=None)
        self.slice = Lambda(self._return_slices, arguments={'channel': int(self.out - self.conv_out_channel)})
        self.concat = Concatenate(axis=-1)

    @staticmethod
    def _return_slices(x, channel):
        y = x[:, :, :, :channel]
        return y

    def call(self, inputs):
        x = self.conv(inputs)
        if self.ratio == 1:
            return x
        dw = self.depthconv(x)
        dw = self.slice(dw)
        x = self.concat([x, dw])
        return x
