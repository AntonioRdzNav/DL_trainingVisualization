from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import math
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import os

batch_size = 32
num_classes = 10
epochs = 25
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_trained_model.h5'
 
# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    print('x_train shape:', trainX.shape)
    print(trainX.shape[0], 'train samples')
    print(testX.shape[0], 'test samples')    

    # Convert class vectors to binary class matrices. 
    trainY = to_categorical(trainY, num_classes)
    testY = to_categorical(testY, num_classes)

    return trainX, trainY, testX, testY
 
# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

def data_augmentation(trainX):
    # This will do preprocessing and realtime data augmentation:	
	datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=False,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=False,  # divide each input by its std
		zca_whitening=False,  # apply ZCA whitening
		zca_epsilon=1e-06,  # epsilon for ZCA whitening
		rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
		# randomly shift images horizontally (fraction of total width)
		width_shift_range=0.1,
		# randomly shift images vertically (fraction of total height)
		height_shift_range=0.1,
		shear_range=0.,  # set range for random shear
		zoom_range=0.,  # set range for random zoom
		channel_shift_range=0.,  # set range for random channel shifts
		# set mode for filling points outside the input boundaries
		fill_mode='nearest',
		cval=0.,  # value used for fill_mode = "constant"
		horizontal_flip=True,  # randomly flip images
		vertical_flip=False,  # randomly flip images
		# set rescaling factor (applied before any other transformation)
		rescale=None,
		# set function that will be applied on each input
		preprocessing_function=None,
		# image data format, either "channels_first" or "channels_last"
		data_format=None,
		# fraction of images reserved for validation (strictly between 0 and 1)
		validation_split=0.0)

	# Compute quantities required for feature-wise normalization
	# (std, mean, and principal components if ZCA whitening is applied).
	datagen.fit(trainX)
	return datagen	
 
# define cnn model
def define_model(trainX):
	model = Sequential()
	model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=trainX.shape[1:]))
	model.add(Conv2D(32, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
	model.add(Conv2D(64, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	# compile model
	# lr = 0.0001
	opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])	
	return model

def printWeightsCL(conLayer):
    x1w = conLayer.get_weights()[0][:,:,0,:]
    for i in range(1,26):
        plt.subplot(5, 5, i)
        plt.imshow(x1w[:,:,i],interpolation="nearest")
    plt.show()    
 
# evaluate a model using k-fold cross-validation
# when to use model.fit?
		# fit is used when the entire training dataset can fit into
		# the memory and no data augmentation is applied.
# when to use model.fit_generator?
		# fit_generator is used when either we have a huge dataset 
		# to fit into our memory or when data augmentation needs to be applied.
def evaluate_model(model, datagen, trainX, trainY, testX, testY):
	lossList, accList = list(), list()
	valLossList, valAccList = list(), list()
	for epoch in range(epochs):
		# printWeightsCL(model.layers[0])
		his = model.fit_generator(
						datagen.flow(trainX, trainY, batch_size=batch_size), 
						initial_epoch=epoch,
						steps_per_epoch=math.ceil(trainX.shape[0] / batch_size),
						epochs=epoch+1,
						validation_data=(testX, testY), 
						validation_steps=None,
						use_multiprocessing=True,
						workers=4, 
						verbose=1
					)
		# printWeightsCL(model.layers[0])
		lossList.append(his.history['loss'])
		accList.append(his.history['acc'])
		valLossList.append(his.history['val_loss'])
		valAccList.append(his.history['val_acc'])
	return lossList, accList, valLossList, valAccList
 
# plot diagnostic learning curves
def summarize_diagnostics(lossList, accList, valLossList, valAccList):
    # plot loss
    plt.subplot(2, 1, 1)
    plt.title('Cross Entropy Loss')
    plt.plot(lossList, color='blue', label='train')
    plt.plot(valLossList, color='orange', label='test')
    # plot accuracy
    plt.subplot(2, 1, 2)
    plt.title('Classification Accuracy')
    # print(history)
    plt.plot(accList, color='blue', label='train')
    plt.plot(valAccList, color='orange', label='test')
    plt.show()
 
# run the test harness for evaluating a model
def main():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# define model
	model = define_model(trainX)	
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# data augmentation
	datagen = data_augmentation(trainX)
	# evaluate model
	lossList, accList, valLossList, valAccList = evaluate_model(model, datagen, trainX, trainY, testX, testY)
	# learning curves
	summarize_diagnostics(lossList, accList, valLossList, valAccList)
 
# entry point, run the test harness
main()