from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, DepthwiseConv2D, Layer, BatchNormalization, Activation

from ghost_bottleneck.components.semodule import SEModule
from ghost_bottleneck.components.ghostmodule import GhostModule


# TODO Finish bottleneck
class GBNeck(Layer):
    def __init__(self, dwkernel, strides, exp, out, ratio, use_se):
        super(Layer, self).__init__()
        self.strides = strides
        self.batchnorm = BatchNormalization()
        self.conv = Conv2D(out, (1,1), strides=(1,1), padding='same', data_format='channels_last',
                           activation=None, use_bias=False)
        self.relu = Activation('relu')
        self.depthconv = DepthwiseConv2D(dwkernel, strides,padding='same',depth_multiplier=ratio-1,
                                         data_format='channels_last', activation=None, use_bias=False)
        self.ghost1 = GhostModule(exp, ratio, 1, 3, use_bias=False)
        self.ghost2 = GhostModule(out, ratio, 1, 3, use_bias=False)

    def call(self, inputs):
        x = self.depthconv(inputs)
        x = self.batchnorm(x)
        x = self.conv(x)
        x = self.batchnorm(x)

        y = self.ghost1(x)
        y = self.batchnorm(y)
        y = self.relu(y)

        if self.strides > 1:
            y = self.depthconv(y)
            y = self.batchnorm(y)




        return x


def GhostBottleneck(x,dwkernel,strides,exp,out,ratio,use_se):
    x1 = DepthwiseConv2D(dwkernel,strides,padding='same',depth_multiplier=ratio-1,data_format='channels_last',
                         activation=None,use_bias=False)(x)
    x1 = BatchNormalization(axis=-1)(x1)
    x1 = Conv2D(out,(1,1),strides=(1,1),padding='same',data_format='channels_last',
               activation=None,use_bias=False)(x1)
    x1 = BatchNormalization(axis=-1)(x1)
    y = GhostModule(x,exp,ratio,1,3)
    y = BatchNormalization(axis=-1)(y)
    y = Activation('relu')(y)
    if(strides>1):
        y = DepthwiseConv2D(dwkernel,strides,padding='same',depth_multiplier=ratio-1,data_format='channels_last',
                         activation=None,use_bias=False)(y)
        y = BatchNormalization(axis=-1)(y)
        y = Activation('relu')(y)
    if(use_se==True):
        y = SEModule(y,exp,ratio)
    y = GhostModule(y,out,ratio,1,3)
    y = BatchNormalization(axis=-1)(y)
    y = add([x1,y])
    return y