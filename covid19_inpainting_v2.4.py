# from inpaint_covid.inpaint_covid import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import tensorflow as tf
from tqdm.notebook import tqdm
from tqdm.keras import TqdmCallback
from tensorflow.keras import backend as K
# from google.colab import drive
import argparse

# parameters
parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument('archi', type=int)
parser.add_argument('ch_init', type=int)
parser.add_argument('lr_value', type=float)
parser.add_argument('LOOP_MASKS', type=int)
parser.add_argument('g_noise', type=float)
parser.add_argument('act_max_value', type=float)
parser.add_argument('act_out_max_value', type=float)
parser.add_argument('NOISE_REDUCTION', type=float)
parser.add_argument('EPOCHS', type=int)
parser.add_argument('EPOCHS_sneak_peek', type=int)
parser.add_argument('LR_REDUCE', type=float)
parser.add_argument('version', type=str)
args = parser.parse_args()

filename = args.filename
archi = args.archi
ch_init = args.ch_init
lr_value = args.lr_value
LOOP_MASKS = args.LOOP_MASKS
g_noise = args.g_noise
act_max_value = args.act_max_value
act_out_max_value = args.act_out_max_value
NOISE_REDUCTION = args.NOISE_REDUCTION
EPOCHS = args.EPOCHS
EPOCHS_sneak_peek = args.EPOCHS_sneak_peek
LR_REDUCE = args.LR_REDUCE
version = args.version

print(filename, archi, ch_init)

# read files
drive.mount('/content/drive')
path_source = '/content/drive/My Drive/Datasets/covid19/COVID-19-20_v2/'
path_dest = '/content/drive/My Drive/KCL/covid19/inpainting_results/'

# read and preprocess
ct, ct_mask, ct_seg = read_covid_CT_and_mask(path_source, filename)
ct, ct_mask, ct_seg = normalize_rotate(ct, ct_mask, ct_seg)
labelled, nr = label(ct_seg>0)
largest_component = (labelled == (np.bincount(labelled.flat)[1:].argmax() + 1))
ct_small, ct_mask_small, ct_seg_small = pad_volume_to_multiple_32(largest_component, ct, ct_mask, ct_seg)

assert 1==2

#this part apparently is not used
labelled, nr = label(ct_seg_small>0)
largest_component = (labelled == (np.bincount(labelled.flat)[1:].argmax() + 1))

#get the masks
bkgd = ct_seg_small == 0
target_mask = np.logical_or(bkgd,ct_mask_small)
target_mask = ~target_mask
target_mask2 = ~bkgd
target_mask3 = ct_mask_small.astype(bool)
mask_target = np.expand_dims(target_mask,0) 
mask_target2 = np.expand_dims(target_mask2,0)
mask_target3 = np.expand_dims(target_mask3,0)

# noise
input_noise = np.random.rand(np.shape(ct_small)[0] ,np.shape(ct_small)[1], np.shape(ct_small)[2])
input_noise = np.expand_dims(input_noise,0) * NOISE_REDUCTION
target = np.expand_dims(ct_small,0)

# initialize
results_all = []
predicted_all = []
epochs_saved = [0]
previous_epochs = 0
model = get_architecture(ct_small, archi, ch_init, g_noise, act_max_value, act_out_max_value)
opt = tf.keras.optimizers.Adam(lr_value) 
loss_masked, mask_used = choose_loss(mask_target, mask_target2, mask_target3, LOSS_USED=0)
model.compile(optimizer=opt, loss=loss_masked)

# Train model
for i in tqdm(range(LOOP_MASKS)):
    results = model.fit(input_noise, target,  epochs=EPOCHS, verbose=0, callbacks=[TqdmCallback(verbose=0)]);
    results_all.extend(results.history['loss'])
    predicted_all.append(model.predict(input_noise)[0,...])
    epochs_saved.append(epochs_saved[-1] + EPOCHS)
    # sneak peek
    loss_masked, mask_used = choose_loss(mask_target, mask_target2, mask_target3, LOSS_USED=2)
    results = model.fit(input_noise, target,  epochs=EPOCHS_sneak_peek, verbose=0, callbacks=[TqdmCallback(verbose=0)]);
    loss_masked, mask_used = choose_loss(mask_target, mask_target2, mask_target3, LOSS_USED=0)
    results_all.extend(results.history['loss'])
    predicted_all.append(model.predict(input_noise)[0,...])
    epochs_sneak_peak = epochs_saved[-1] + EPOCHS_sneak_peek
    epochs_saved.append(epochs_sneak_peak)

    lr_value = lr_value * LR_REDUCE
    K.set_value(model.optimizer.learning_rate, lr_value)

# make figure and save it
SLICE = 100
plot_inpaints_pairs(np.asarray(predicted_all)[...,SLICE], epochs_saved, target[0,...,SLICE], mask_used, mask_target3[0,...,SLICE], results_all, parameters, blend='blend', slice_mask=SLICE, path_dest=path_dest, save=True, version=version)


