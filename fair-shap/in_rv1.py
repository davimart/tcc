"""
Reference: https://github.com/vQuadX/face-recognition-server
Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning - https://arxiv.org/abs/1602.07261
"""


from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, ReLU, BatchNormalization, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Lambda, Concatenate, Add, Activation, GlobalAveragePooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


def myconv2d(x, filters, kernel_size, strides=1, padding='same', activation='relu', use_bias=False, name=None):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, name=name)(x)
    if not use_bias:
        normalization_axis = 1 if K.image_data_format() == 'channels_first' else 3
        x = BatchNormalization(
            axis=normalization_axis,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name=f'{name}_BatchNorm' if name else None
        )(x)
    if activation is not None:
        x = Activation(activation, name=f'{name}_Activation' if name else None)(x)
    return x

def resnet_block(x, branches, scale, activation, prefix):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    mixed = Concatenate(axis=channel_axis, name=f'{prefix}_Concatenate')(branches)
    up = myconv2d(mixed, K.int_shape(x)[channel_axis], 1, activation=None, use_bias=True,
                        name=f'{prefix}_Conv2d_1x1')

    up = Lambda(lambda _up: _up * scale, output_shape=K.int_shape(up)[1:])(up)
    x = Add()([x, up])
    if activation is not None:
        x = Activation(activation, name=f'{prefix}_Activation')(x)
    return x

def stem(input_x):
    """Builds the Stem of the Inception-ResNet network"""
    #inputs = Input(shape=_input_shape)
    # 149 x 149 x 32
    x = myconv2d(input_x, 32, 3, strides=2, padding='valid', name='Conv2d_1a_3x3')
    # 147 x 147 x 32
    x = myconv2d(x, 32, 3, padding='valid', name='Conv2d_2a_3x3')
    # 147 x 147 x 64
    x = myconv2d(x, 64, 3, name='Conv2d_2b_3x3')
    # 73 x 73 x 64
    x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)
    # 73 x 73 x 80
    x = myconv2d(x, 80, 1, padding='valid', name='Conv2d_3b_1x1')
    # 71 x 71 x 192
    x = myconv2d(x, 192, 3, padding='valid', name='Conv2d_4a_3x3')
    # 35 x 35 x 256
    x = myconv2d(x, 256, 3, strides=2, padding='valid', name='Conv2d_4b_3x3')
    return input_x, x

def block35(x, block_idx, scale=1.0, activation='relu'):
    """Builds the 35x35 Inception-ResNet-A module"""
    prefix = f'Block35_{block_idx}' if block_idx is not None else None

    branch_0 = myconv2d(x, 32, 1, name=f'{prefix}_Branch_0_Conv2d_1x1')
    branch_1 = myconv2d(x, 32, 1, name=f'{prefix}_Branch_1_Conv2d_0a_1x1')
    branch_1 = myconv2d(branch_1, 32, 3, name=f'{prefix}_Branch_1_Conv2d_0b_3x3')
    branch_2 = myconv2d(x, 32, 1, name=f'{prefix}_Branch_2_Conv2d_0a_1x1')
    branch_2 = myconv2d(branch_2, 32, 3, name=f'{prefix}_Branch_2_Conv2d_0b_3x3')
    branch_2 = myconv2d(branch_2, 32, 3, name=f'{prefix}_Branch_2_Conv2d_0c_3x3')
    branches = [branch_0, branch_1, branch_2]

    return resnet_block(x, branches, scale, activation, prefix)

def reduction_a( x, k=192, l=192, m=256, n=384):
    """Builds the 35x35 to 17x17 Reduction-A module"""
    branch_0 = myconv2d(x, n, 3, strides=2, padding='valid', name=f'Mixed_6a_Branch_0_Conv2d_1a_3x3')
    branch_1 = myconv2d(x, k, 1, name=f'Mixed_6a_Branch_1_Conv2d_0a_1x1')
    branch_1 = myconv2d(branch_1, l, 3, name=f'Mixed_6a_Branch_1_Conv2d_0b_3x3')
    branch_1 = myconv2d(branch_1, m, 3, strides=2, padding='valid', name=f'Mixed_6a_Branch_1_Conv2d_1a_3x3')
    branch_pool = MaxPooling2D(3, strides=2, padding='valid', name=f'Mixed_6a_Branch_2_MaxPool_1a_3x3')(x)
    branches = [branch_0, branch_1, branch_pool]
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    return Concatenate(axis=channel_axis, name='Mixed_6a')(branches)

def reduction_b( x):
    """Builds the 17x17 to 8x8 Reduction-B module"""
    branch_0 = myconv2d(x, 256, 1, name=f'Mixed_7a_Branch_0_Conv2d_0a_1x1')
    branch_0 = myconv2d(branch_0, 384, 3, strides=2, padding='valid', name=f'Mixed_7a_Branch_0_Conv2d_1a_3x3')
    branch_1 = myconv2d(x, 256, 1, name=f'Mixed_7a_Branch_1_Conv2d_0a_1x1')
    branch_1 = myconv2d(branch_1, 256, 3, strides=2, padding='valid', name=f'Mixed_7a_Branch_1_Conv2d_1a_3x3')
    branch_2 = myconv2d(x, 256, 1, name=f'Mixed_7a_Branch_2_Conv2d_0a_1x1')
    branch_2 = myconv2d(branch_2, 256, 3, name=f'Mixed_7a_Branch_2_Conv2d_0b_3x3')
    branch_2 = myconv2d(branch_2, 256, 3, strides=2, padding='valid', name=f'Mixed_7a_Branch_2_Conv2d_1a_3x3')
    branch_3_pool = MaxPooling2D(3, strides=2, padding='valid', name=f'Mixed_7a_Branch_3_MaxPool_1a_3x3')(x)
    branches = [branch_0, branch_1, branch_2, branch_3_pool]
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    return Concatenate(axis=channel_axis, name='Mixed_7a')(branches)

def block17( x, block_idx, scale=1.0, activation='relu'):
    """Builds the 17x17 Inception-ResNet-B module"""
    prefix = f'Block17_{block_idx}' if block_idx is not None else None

    branch_0 = myconv2d(x, 128, 1, name=f'{prefix}_Branch_0_Conv2d_1x1')
    branch_1 = myconv2d(x, 128, 1, name=f'{prefix}_Branch_1_Conv2d_0a_1x1')
    branch_1 = myconv2d(branch_1, 128, [1, 7], name=f'{prefix}_Branch_1_Conv2d_0b_1x7')
    branch_1 = myconv2d(branch_1, 128, [7, 1], name=f'{prefix}_Branch_1_Conv2d_0c_7x1')
    branches = [branch_0, branch_1]

    return resnet_block(x, branches, scale, activation, prefix)

def block8( x, block_idx, scale=1.0, activation='relu'):
    """Builds the 8x8 Inception-ResNet-C module"""
    prefix = f'Block8_{block_idx}' if block_idx is not None else None

    branch_0 = myconv2d(x, 192, 1, name=f'{prefix}_Branch_0_Conv2d_1x1')
    branch_1 = myconv2d(x, 192, 1, name=f'{prefix}_Branch_1_Conv2d_0a_1x1')
    branch_1 = myconv2d(branch_1, 192, [1, 3], name=f'{prefix}_Branch_1_Conv2d_0b_1x3')
    branch_1 = myconv2d(branch_1, 192, [3, 1], name=f'{prefix}_Branch_1_Conv2d_0c_3x1')
    branches = [branch_0, branch_1]

    return resnet_block(x, branches, scale, activation, prefix)


def build_inception_rn_v1(input_shape = (160,160,3), batch_size=128, out_dim=1, dropout_keep_prob=0.8):

    #INPUT
    input_img = Input(shape = input_shape) #160x160x128 batch_size=batch_size
    inputs, x = stem(input_img)

    # 5 x Inception-ResNet-A module
    for block_idx in range(1, 6):
        x = block35(x, block_idx, scale=0.17)

    # Reduction-A module
    x = reduction_a(x)

    # 10 x Inception-ResNet-B module
    for block_idx in range(1, 11):
        x = block17(x, block_idx, scale=0.1)

    # Reduction-B module
    x = reduction_b(x)

    # 5 x Inception-ResNet-C module
    for block_idx in range(1, 6):
        x = block8(x, block_idx, scale=0.2)
    x = block8(x, block_idx=6, activation=None)

    #Classification top layers
    x = GlobalAveragePooling2D(name='AvgPool')(x)
    x = Dropout(1.0 - dropout_keep_prob, name='Dropout')(x)

    x = Flatten()(x)

    # Bottleneck
    x = Dense(224, name='mlp_1')(x)
    x = BatchNormalization(name='BN_mlp_1')(x)
    x = Activation('relu', name=f'relu_mlp_1')(x)

    x = Dense(56, name='mlp_2')(x)
    x = BatchNormalization(name='BN_mlp_2')(x)
    x = Activation('relu', name=f'relu_mlp_2')(x)

    x = Dense(out_dim,
                activation='sigmoid',
                #use_bias=False,
                name='output')(x)
    #x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name='Bottleneck_BatchNorm')(x)

    return Model(input_img, x)

if __name__=="__main__":
    model = build_inception_rn_v1()
    model.summary()