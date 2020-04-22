from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, Lambda, Reshape, Layer, Activation


class SEModule(Layer):
    def __init__(self, filters, ratio):
        super(SEModule, self).__init__()
        self.pooling = GlobalAveragePooling2D(data_format='channels_last')
        self.reshape = Lambda(self._reshape)
        self.conv = Conv2D(int(filters / ratio), (1,1), strides=(1,1), padding='same',
                           data_format='channels_last', use_bias=False, activation=None)
        self.relu = Activation('relu')
        self.hard_sigmoid = Activation('hard_sigmoid')

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
        x = self.conv(x)
        x = self.relu(x)
        excitation = self.conv(x)
        excitation = self.hard_sigmoid(excitation)
        x = Lambda(self.excite, arguments={'excitation': excitation})(inputs)
        return x
