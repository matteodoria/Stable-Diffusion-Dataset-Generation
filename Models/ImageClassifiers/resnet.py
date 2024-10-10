import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
from keras import backend as K

def pad_depth(x, desired_channels):
    padding = desired_channels - tf.shape(x)[-1]
    zero_padding = tf.zeros_like(x)[:, :, :, :padding]
    return tf.concat([x, zero_padding], axis=-1)

def BasicBlock(x, filters, strides=1, l2=2e-4, seed=42, name=''):
    input_dim = K.int_shape(x)[-1]
    skip = x

    # First convolution layer
    x = layers.Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same',
                      use_bias=False, kernel_initializer=keras.initializers.HeNormal(seed),
                      kernel_regularizer=keras.regularizers.L2(l2), name=name + 'conv1')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=name + 'bn1')(x)
    x = layers.ReLU(name=name + 'relu1')(x)

    # Second convolution layer
    x = layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
                      use_bias=False, kernel_initializer=keras.initializers.HeNormal(seed),
                      kernel_regularizer=keras.regularizers.L2(l2), name=name + 'conv2')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=name + 'bn2')(x)

    # Adjusting skip connection
    if strides != 1 or input_dim != filters:
        skip = layers.MaxPooling2D(pool_size=(3, 3), strides=strides, padding='same', name=name + 'pooling')(skip)
        skip = pad_depth(skip, filters)

    # Combining main path and shortcut
    x = layers.Add(name=name + 'add')([x, skip])
    x = layers.ReLU(name=name + 'relu2')(x)

    return x

class ResNet(Model):
    def __init__(self, input_shape, block, num_blocks, num_classes, initial_filters=16, l2=0, seed=42, name=''):
        input_layer = layers.Input(shape=input_shape, name='input_layer')

        x = layers.Conv2D(filters=initial_filters, kernel_size=3, strides=1, padding='same', use_bias=False,
                          kernel_initializer=keras.initializers.HeNormal(seed),
                          kernel_regularizer=keras.regularizers.L2(l2),
                          name='conv0')(input_layer)

        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn0')(x)
        x = layers.ReLU(name='relu0')(x)

        for i, num_block in enumerate(num_blocks):
            filters = initial_filters * 2 ** i
            for j in range(num_block):
                strides = 2 if i > 0 and j == 0 else 1
                x = block(x, filters, strides, l2=l2, seed=seed, name=f'block{i}{j}_')

        x = layers.GlobalAveragePooling2D(name='gap')(x)
        output_layer = layers.Dense(num_classes, activation='softmax',
                                    kernel_initializer=keras.initializers.GlorotNormal(seed),
                                    kernel_regularizer=keras.regularizers.L2(l2), name='output_layer')(x)

        super().__init__(input_layer, output_layer, name=name)

class ResNet20(ResNet):
    def __init__(self, input_shape, num_classes, initial_filters=16, seed=1234):
        super().__init__(input_shape = input_shape,
                         block = BasicBlock,
                         num_blocks = [3, 3, 3],
                         num_classes = num_classes,
                         initial_filters = initial_filters,
                         seed = seed,
                         name = "ResNet20")
