import tensorflow as tf
import visualkeras
from PIL import ImageFont
from keras_visualizer import visualizer
from collections import defaultdict
from model import BatchNorm, BilinearUpsampling, Repeat
from tensorflow.python.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Concatenate, Activation, \
    GlobalAveragePooling2D, Dense, Reshape, Multiply, Add

# Dependencies:
#
# visualkeras = "^0.0.2"
# keras-visualizer = "^3.1.2"
# pydot = "^1.4.2"
#
# visualkeras needs to be modded to work

model = None  # obtain model from somewhere


# 1
tf.keras.utils.plot_model(model)


# 2
font = ImageFont.truetype("arial.ttf", 32)
color_map = defaultdict(dict)
color_map[InputLayer.__name__]['fill'] = (47, 79, 79, 255)
color_map[Reshape.__name__]['fill'] = (128, 0, 0, 255)
color_map[MaxPooling2D.__name__]['fill'] = (0, 0, 100, 255)
color_map[Dropout.__name__]['fill'] = (0, 128, 0, 255)
color_map[Concatenate.__name__]['fill'] = (255, 0, 140, 255)
color_map[Activation.__name__]['fill'] = (222, 135, 184, 255)
color_map[GlobalAveragePooling2D.__name__]['fill'] = (0, 0, 255, 255)
color_map[Dense.__name__]['fill'] = (0, 255, 191, 255)
color_map[Conv2D.__name__]['fill'] = (0, 255, 0, 255)
color_map[Multiply.__name__]['fill'] = (255, 255, 0, 255)
color_map[Add.__name__]['fill'] = (255, 84, 255, 255)
color_map[BatchNorm.__name__]['fill'] = (221, 221, 160, 255)
color_map[BilinearUpsampling.__name__]['fill'] = (255, 147, 20, 255)
color_map[Repeat.__name__]['fill'] = (127, 212, 255, 255)

visualkeras.layered_view(model, legend=True, font=font, color_map=color_map, draw_volume=False).show()


# 3
visualizer(model, file_format="png")

