# AUTOGENERATED! DO NOT EDIT! File to edit: 01_unets.ipynb (unless otherwise specified).

__all__ = ['unet5']

# Cell
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GaussianNoise
from scipy.ndimage import binary_erosion, binary_dilation
from tensorflow.keras.layers import LeakyReLU, ReLU
from tqdm.keras import TqdmCallback
from tensorflow.keras import backend as K

# Cell
def unet5(ct_small, ch=32, g_noise= 0.3, act_max_value = 1, act_out_max_value = 1):
    IMG_CHANNELS = np.shape(ct_small)[-1]
    inputs = Input(np.shape(ct_small))
    c1 = Conv2D(ch, (3, 3), kernel_initializer='he_normal', padding='same') (inputs)
    if (g_noise > 0): c1 = GaussianNoise(g_noise) (c1)
    c1 = BatchNormalization()(c1)
    c1 = ReLU(max_value=act_max_value)(c1)
    # c1 = Dropout(0.1) (c1)
    c1 = Conv2D(ch, (3, 3), kernel_initializer='he_normal', padding='same') (c1)
    if (g_noise > 0): c1 = GaussianNoise(g_noise) (c1)
    c1 = BatchNormalization()(c1)
    c1 = ReLU(max_value=act_max_value)(c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(ch*2, (3, 3), kernel_initializer='he_normal', padding='same') (p1)
    if (g_noise > 0): c2 = GaussianNoise(g_noise) (c2)
    c2 = BatchNormalization()(c2)
    c2 = ReLU(max_value=act_max_value)(c2)
    # c2 = Dropout(0.1) (c2)
    c2 = Conv2D(ch*2, (3, 3), kernel_initializer='he_normal', padding='same') (c2)
    if (g_noise > 0): c2 = GaussianNoise(g_noise) (c2)
    c2 = BatchNormalization()(c2)
    c2 = ReLU(max_value=act_max_value)(c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(ch*4, (3, 3), kernel_initializer='he_normal', padding='same') (p2)
    if (g_noise > 0): c3 = GaussianNoise(g_noise) (c3)
    c3 = BatchNormalization()(c3)
    c3 = ReLU(max_value=act_max_value)(c3)
    # c3 = Dropout(0.2) (c3)
    c3 = Conv2D(ch*4, (3, 3), kernel_initializer='he_normal', padding='same') (c3)
    if (g_noise > 0): c3 = GaussianNoise(g_noise) (c3)
    c3 = BatchNormalization()(c3)
    c3 = ReLU(max_value=act_max_value)(c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(ch*8, (3, 3), kernel_initializer='he_normal', padding='same') (p3)
    if (g_noise > 0): c4 = GaussianNoise(g_noise) (c4)
    c4 = BatchNormalization()(c4)
    c4 = ReLU(max_value=act_max_value)(c4)
    # c4 = Dropout(0.2) (c4)
    c4 = Conv2D(ch*8, (3, 3), kernel_initializer='he_normal', padding='same') (c4)
    if (g_noise > 0): c4 = GaussianNoise(g_noise) (c4)
    c4 = BatchNormalization()(c4)
    c4 = ReLU(max_value=act_max_value)(c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(ch*16, (3, 3), kernel_initializer='he_normal', padding='same') (p4)
    if (g_noise > 0): c5 = GaussianNoise(g_noise) (c5)
    c5 = BatchNormalization()(c5)
    c5 = ReLU(max_value=act_max_value)(c5)
    # c5 = Dropout(0.3) (c5)
    c5 = Conv2D(ch*16, (3, 3), kernel_initializer='he_normal', padding='same') (c5)
    if (g_noise > 0): c5 = GaussianNoise(g_noise) (c5)
    c5 = BatchNormalization()(c5)
    c5 = ReLU(max_value=act_max_value)(c5)

    out_inter = c5

    u6 = Conv2DTranspose(ch*4, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(ch*8, (3, 3), kernel_initializer='he_normal', padding='same') (u6)
    if (g_noise > 0): c6 = GaussianNoise(g_noise) (c6)
    c6 = BatchNormalization()(c6)
    c6 = ReLU(max_value=act_max_value)(c6)
    # c6 = Dropout(0.2) (c6)
    c6 = Conv2D(ch*8, (3, 3), kernel_initializer='he_normal', padding='same') (c6)
    if (g_noise > 0): c6 = GaussianNoise(g_noise) (c6)
    c6 = BatchNormalization()(c6)
    c6 = ReLU(max_value=act_max_value)(c6)

    u7 = Conv2DTranspose(ch*2, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(ch*4, (3, 3), kernel_initializer='he_normal', padding='same') (u7)
    if (g_noise > 0): c7 = GaussianNoise(g_noise) (c7)
    c7 = BatchNormalization()(c7)
    c7 = ReLU(max_value=act_max_value)(c7)
    # c7 = Dropout(0.2) (c7)
    c7 = Conv2D(ch*4, (3, 3), kernel_initializer='he_normal', padding='same') (c7)
    if (g_noise > 0): c7 = GaussianNoise(g_noise) (c7)
    c7 = BatchNormalization()(c7)
    c7 = ReLU(max_value=act_max_value)(c7)

    u8 = Conv2DTranspose(ch, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(ch*2, (3, 3), kernel_initializer='he_normal', padding='same') (u8)
    if (g_noise > 0): c8 = GaussianNoise(g_noise) (c8)
    c8 = BatchNormalization()(c8)
    c8 = ReLU(max_value=act_max_value)(c8)
    # c8 = Dropout(0.1) (c8)
    c8 = Conv2D(ch*2, (3, 3), kernel_initializer='he_normal', padding='same') (c8)
    if (g_noise > 0): c8 = GaussianNoise(g_noise) (c8)
    c8 = BatchNormalization()(c8)
    c8 = ReLU(max_value=act_max_value)(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(ch, (3, 3), kernel_initializer='he_normal', padding='same') (u9)
    if (g_noise > 0): c9 = GaussianNoise(g_noise) (c9)
    c9 = BatchNormalization()(c9)
    c9 = ReLU(max_value=act_max_value)(c9)
    # c9 = Dropout(0.1) (c9)
    c9 = Conv2D(ch, (3, 3), kernel_initializer='he_normal', padding='same') (c9)
    if (g_noise > 0): c9 = GaussianNoise(g_noise) (c9)
    c9 = BatchNormalization()(c9)
    c9 = ReLU(max_value=act_max_value)(c9)

    outputs = Conv2D(IMG_CHANNELS, (1, 1)) (c9)
    outputs = ReLU(max_value=act_out_max_value)(outputs)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model