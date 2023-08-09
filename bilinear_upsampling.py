from tensorflow.python.keras.utils import conv_utils
from keras.layers import Layer, InputSpec
import tensorflow as tf


class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, input_size=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.input_size = input_size
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                     input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                    input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs, **kwargs):
        if self.upsampling:
            # print(self.name, inputs.shape, self.input_size)

            i1 = inputs.shape[1]
            i2 = inputs.shape[2]

            if i1 is None or i2 is None:
                i1 = self.input_size[0]
                i2 = self.input_size[1]

            return tf.compat.v1.image.resize(
                inputs,
                (int(i1 * self.upsampling[0]), int(i2 * self.upsampling[1])),
                align_corners=True,
                method=tf.compat.v1.image.ResizeMethod.BILINEAR
            )
        else:
            return tf.compat.v1.image.resize(
                inputs,
                (self.output_size[0], self.output_size[1]),
                align_corners=True,
                method=tf.compat.v1.image.ResizeMethod.BILINEAR
            )

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
