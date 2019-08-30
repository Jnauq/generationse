from __future__ import division
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

import keras.callbacks
import keras.backend as K

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# Displays the current learning rate at each epoch
class AdamLearningRateTracker(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = float(K.get_value(self.model.optimizer.lr))
        print("Learning rate:", lr)


# Helper funtion for step decay of learning rate
def step_decay(epoch):
    lr = 0.001
    drop = 0.5
    epochs_drop = 50.0
    learnrate = lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return learnrate


# original directory with spect data
train_dir = '/home/justin/PycharmProjects/Machine/trialcvrgb/train'
val_dir = '/home/justin/PycharmProjects/Machine/trialcvrgb/val'
test_dir = '/home/justin/PycharmProjects/Machine/trialcvrgb/test'
fig_dir = '/home/justin/PycharmProjects/Machine/trialcvrgb'

# constants
num_train_samples = sum([len(files) for r, d, files in os.walk(train_dir)])
num_val_samples = sum([len(files) for r, d, files in os.walk(val_dir)])
num_test_samples = sum([len(files) for r, d, files in os.walk(test_dir)])
epochs = 500
batch_size = 16
img_height = 109
img_width = 91
lrate = 0.001


# build pretrained model inceptionV3 using imagenet weights
base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))


# build own classifier model to add on to base_model
model_top = Sequential()
model_top.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:], data_format=None))
model_top.add(Dense(256, activation='relu'))
model_top.add(Dropout(0.2))
model_top.add(Dense(1, activation='sigmoid'))
model = Model(inputs=base_model.input, outputs=model_top(base_model.output))


# compile model
model.compile(optimizer=Adam(lr=lrate), loss='binary_crossentropy', metrics=['acc'])

# image augmentations
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # shear_range=0.01,
    # rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.75, 1.25],
    horizontal_flip=True
)


test_datagen = ImageDataGenerator(
    rescale=1./255
)


# generator randomly shuffles and presents images in batches to the network
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=1,
    class_mode="binary",
    shuffle=False
)


# callbacks
# es = EarlyStopping(monitor='val_acc', min_delta=0, patience=8, verbose=0, mode='auto')
# filepath = './models/model-best.hdf5'
# cp = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=0,
#                      save_best_only=True, save_weights_only=False, mode='auto')
lrs = LearningRateScheduler(step_decay)
alrt = AdamLearningRateTracker()


# Fine-tune pretrained model using data generator
hist = model.fit_generator(
    train_generator,
    steps_per_epoch=num_train_samples//batch_size,
    epochs=epochs,
    callbacks=[lrs, alrt]
)


# # plotting accuracy and loss plots to visualize if overfitting is occurring
# accuracy = hist.history['acc']
# val_accuracy = hist.history['val_acc']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# epochs = range(len(accuracy))
#
# # summarize accuracy
# plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
# plt.plot(epochs, val_accuracy, 'g', label='Validation Accuracy')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.xticks(np.arange(min(epochs), max(epochs)+1, 2))
# plt.legend()
# plt.savefig(fig_dir + '/accuracy.png', bbox_inches='tight', pad_inches=0)
# plt.close(fig=None)
#
# # summarize loss
# plt.plot(epochs, loss, 'b', label='Training Loss')
# plt.plot(epochs, val_loss, 'g', label='Validation Loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.xticks(np.arange(min(epochs), max(epochs)+1, 2))
# plt.legend()
# plt.savefig(fig_dir + '/loss.png', bbox_inches='tight', pad_inches=0)
# plt.close(fig=None)


# # load best saved model
# model = load_model('./models/model-best.hdf5')
# # compile loaded model
# model.compile(optimizer=Adam(lr=lrate), loss='binary_crossentropy', metrics=['acc'])


# evaluating the saved model
test_generator.reset()
results = model.evaluate_generator(test_generator, steps=num_test_samples)
print('\nTest accuracy: %.4f' % results[1])
print('Test loss: %.4f' % results[0])


# plotting the precision-recall curve
test_generator.reset()
probabilities = model.predict_generator(test_generator, steps=num_test_samples)
predicted_class_indices = [1 if x > 0.5 else 0 for x in probabilities]

labels = test_generator.class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames = test_generator.filenames
probabilities = [np.asscalar(p) for p in probabilities]

# results = pd.DataFrame({"Filename":filenames, "Predictions":predictions, "Probabilities":probabilities})
# results.to_csv("/home/justin/PycharmProjects/Machine/trialcvrgb/results.csv", index=True)


ytest = []
for file in filenames:
    if 'Control' in file:
        ytest.append(0)
    else:
        ytest.append(1)

ytest = np.asarray(ytest)

# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(ytest, probabilities)
auc = auc(recall, precision)
print('PR auc:%.4f' % auc)

fpr, tpr, threshold = roc_curve(ytest, probabilities)
roc_auc = roc_auc_score(ytest, probabilities)
print('ROC auc:%.4f' % roc_auc)

# ROC when there is an equal number of images for each class
plt.plot(fpr, tpr, 'g-')
plt.plot([0, 1], [0, 1], 'r--')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.xticks(np.arange(min(fpr), max(fpr)+0.1, 0.2))
plt.savefig(fig_dir + '/ROC.png', bbox_inches='tight', pad_inches=0)
plt.close(fig=None)

# PR when there is a large difference in number of images between classes
plt.plot(recall, precision, 'b-')
plt.plot([0, 1], [min(precision), min(precision)], 'r--')
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.xticks(np.arange(min(recall), max(recall)+0.1, 0.2))
plt.savefig(fig_dir + '/AUC.png', bbox_inches='tight', pad_inches=0)
plt.close(fig=None)


# Calculate the sensitivity and specificity of test results
tp=0
fp=0
tn=0
fn=0
for i in range(0, len(filenames)):
    if predictions[i] == 'pd':
        if predictions[i] in filenames[i].lower():
            tp += 1
        else:
            fp += 1
    else:
        if predictions[i] in filenames[i].lower():
            tn += 1
        else:
            fn += 1

sensitivity = tp/float(tp + fn)
specificity = tn/float(tn + fp)
print('\nTpos: %d, Fpos: %d, Tneg: %d, Fneg: %d' % (tp, fp, tn, fn))
print('Sensitivity: %f\t Specificity: %f' % (sensitivity, specificity))



