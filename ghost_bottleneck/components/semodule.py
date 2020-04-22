from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, Lambda, Reshape, Layer


class SEModule(Layer):
    def __init__(self, filters, ratio):
        super(Layer, self).__init__()
        self.pooling = GlobalAveragePooling2D(data_format='channels_last')
        self.reshape = Lambda(self._reshape)
        self.conv1 = Conv2D(int(filters / ratio), (1,1), strides=(1,1), padding='same',
                           data_format='channels_last', use_bias=False, activation='relu')
        self.conv2 = Conv2D(int(filters / ratio), (1, 1), strides=(1, 1), padding='same',
                            data_format='channels_last', use_bias=False, activation='hard_sigmoid')

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
