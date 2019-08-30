from __future__ import division
import os
from numpy import mean, std
import matplotlib.pyplot as plt
import numpy as np
import math

import keras.callbacks
import keras.backend as K

from keras import applications
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam

##############################################################
#
# Cross validation performance evaluation of hyperparameters
#
##############################################################


# input directory with folds
inputdir = '/home/justin/PycharmProjects/Machine/cvrgb'

# list of test results
pred_scores = []
table = []

class AdamLearningRateTracker(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = float(K.get_value(self.model.optimizer.lr))
        print("Learning rate:", lr)


def step_decay(epoch):
    lr = 0.001
    drop = 0.5
    epochs_drop = 50.0
    learnrate = lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return learnrate


for i in range(1, 11):

    curdir = os.path.join(inputdir, 'fold{}'.format(i))
    traindir = os.path.join(curdir, 'train')
    # testdir = os.path.join(curdir, 'test')
    valdir = os.path.join(curdir, 'val')

    # constants
    num_train_samples = sum([len(files) for r, d, files in os.walk(traindir)])
    # num_test_samples = sum([len(files) for r, d, files in os.walk(testdir)])
    num_val_samples = sum([len(files) for r, d, files in os.walk(valdir)])
    epochs = 500
    batch_size = 16
    img_height = 109
    img_width = 91
    lrate = 0.001

    # callbacks
    lrs = LearningRateScheduler(step_decay)
    alrt = AdamLearningRateTracker()

    # build pre-trained model inceptionV3 using imagenet weights
    base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    # build own classifier model to add on to base_model
    model_top = Sequential()
    model_top.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:]))
    model_top.add(Dense(256, activation='relu'))
    model_top.add(Dropout(0.2))
    model_top.add(Dense(1, activation='sigmoid'))
    model = Model(inputs=base_model.input, outputs=model_top(base_model.output))

    # compile model
    model.compile(optimizer=Adam(lr=lrate), loss='binary_crossentropy', metrics=['acc'])

    # image augmentations
    train_datagen = ImageDataGenerator(
        rescale=1. / 255.0,
        # shear_range=0.01,
        # rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.75, 1.25],
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(
        rescale=1. / 255.0
    )

    # test_datagen = ImageDataGenerator(
    #     rescale=1. / 255.0
    # )

    # generator randomly shuffles and presents images in batches to the network
    train_generator = train_datagen.flow_from_directory(
        traindir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        valdir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True
    )

    # test_generator = test_datagen.flow_from_directory(
    #     testdir,
    #     target_size=(img_height, img_width),
    #     batch_size=1,
    #     class_mode="binary",
    #     shuffle=False
    # )

    print('\nTRAINING ON FOLD {}\n'.format(i))

    # Fine-tune pretrained model using data generator
    hist = model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_samples//batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=num_val_samples//batch_size,
        callbacks=[lrs, alrt]
    )


    # plotting accuracy and loss plots to visualize if overfitting is occurring
    accuracy = hist.history['acc']
    val_accuracy = hist.history['val_acc']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(len(accuracy))

    # summarize accuracy
    plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
    plt.plot(epochs, loss, 'c', label='Training Loss')
    plt.plot(epochs, val_loss, 'y', label='Validation Loss')
    plt.ylim([0.0, 1.1])
    plt.xlabel('Epochs')
    # plt.xticks(np.arange(min(epochs), max(epochs)+2, 2))
    plt.xticks([], [])
    plt.legend()
    plt.savefig(inputdir + '/fold{}.png'.format(i), bbox_inches='tight', pad_inches=0)
    plt.close(fig=None)


#     # evaluating the saved model
#     test_generator.reset()
#     scores = model.evaluate_generator(test_generator, steps=num_test_samples)
#     print('\nTest accuracy: %.4f' % scores[1])
#     print('Test loss: %.4f' % scores[0])
#     pred_scores.append((scores[1], num_test_samples))
#
#
#     # predicting test images
#     test_generator.reset()
#     probs = model.predict_generator(test_generator, steps=num_test_samples)
#     predicted_class = [1 if x > 0.5 else 0 for x in probs]
#
#     labels = test_generator.class_indices
#     labels = dict((v,k) for k,v in labels.items())
#     predictions = [labels[k] for k in predicted_class]
#
#     filenames = test_generator.filenames
#
#     tp=0
#     fp=0
#     tn=0
#     fn=0
#     for j in range(0, len(filenames)):
#         if predictions[j] == 'pd':
#             if predictions[j] in filenames[j].lower():
#                 tp += 1
#             else:
#                 fp += 1
#         else:
#             if predictions[j] in filenames[j].lower():
#                 tn += 1
#             else:
#                 fn += 1
#     sensitivity = tp/float(tp + fn)
#     specificity = tn/float(tn + fp)
#     table.append(('Fold {}:'.format(i), 'tp='+str(tp), 'fp='+str(fp), 'tn='+str(tn), 'fn='+str(fn), 'sen='+str(sensitivity), 'spec='+str(specificity)))
#     print('Sensitivity: %f\t Specificity: %f' % (sensitivity, specificity))
#
#
# total = 0.0
# count = 0.0
# for idx, item in enumerate(pred_scores):
#     print('Fold {} Test Accuracy: %.4f'.format(idx+1) % item[0])
#     total += (item[0]*item[1])
#     count += item[1]
# print('Weighted Average Test Accuracy: %.4f (+/- %0.4f)' % (total/count, (std([item[0] for item in pred_scores]))))
# # print('Mean Test Accuracy: %.4f (+/- %0.4f)' % (mean([item[0] for item in pred_scores]), std([item[0] for item in pred_scores])))
# for result in table:
#     print(result)
