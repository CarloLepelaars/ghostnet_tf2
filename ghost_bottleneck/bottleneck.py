from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, DepthwiseConv2D, Lambda, Reshape, Layer

# TODO Finish bottleneck
class GBNeck(Layer):
    def __init__(self, dwkernel, strides, exp, out, ratio, use_se):
        super(Layer, self).__init__()
        self.pooling = GlobalAveragePooling2D(data_format='channels_last')
        self.reshape = Lambda(self._reshape)
        self.conv1 = Conv2D(int(out / ratio), (1, 1), strides=(1, 1), padding='same',
                           data_format='channels_last', use_bias=False, activation='relu')
        self.depthconv = DepthwiseConv2D(dwkernel,strides,padding='same',depth_multiplier=ratio-1,
                                         data_format='channels_last', activation=None, use_bias=False)

    @staticmethod
    def _reshape(x):
        y = Reshape((1, 1, int(x.shape[1])))
        return y

    @staticmethod
    def _excite(x, excitation):
        return x * excitation

    def call(self, inputs):
        x = self.pooling(inputs)
        x = self.reshape(x)
        x = self.conv1(x)
        excitation = self.conv2(x)
        x = Lambda(self.excite, arguments={'excitation': excitation})(inputs)
        return x
