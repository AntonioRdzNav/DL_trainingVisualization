# baseline cnn model for mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import SGD
 
# load train and test dataset
def load_dataset():
    # load dataset (70,000 images in total)
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX[:20000], trainY[:20000], testX[:3000], testY[:3000]
 
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
 
# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='sigmoid', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='sigmoid', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def printWeightsCL(conLayer):
    x1w = conLayer.get_weights()[0][:,:,0,:]
    for i in range(1,10):
        plt.subplot(3,3,i)
        plt.imshow(x1w[:,:,i],interpolation="nearest",cmap="gray")
    plt.show()    
 
# evaluate a model using k-fold cross-validation
def evaluate_model(trainX, trainY, testX, testY, epochs=10):
    history, score = list(), list()
    for epoch in range(epochs):
        # define model
        model = define_model()
        printWeightsCL(model.layers[0])
        # fit model
        his = model.fit(trainX, trainY, epochs=1, batch_size=32, validation_data=(testX, testY), verbose=1)
        history.append(his)
        printWeightsCL(model.layers[0])
        # evaluate model
        _, sco = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (sco * 100.0))
        score.append(sco)
    return score, history
 
# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    plt.subplot(2, 1, 1)
    print(history.history['loss'])
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    plt.subplot(2, 1, 2)
    plt.title('Classification Accuracy')
    # print(history)
    plt.plot(history.history['acc'], color='blue', label='train')
    plt.plot(history.history['val_acc'], color='orange', label='test')
    plt.show()
 
# summarize model performance
def summarize_performance(score):
	# print summary
	print('Accuracy: %.3f' % score)
 
# run the test harness for evaluating a model
def main():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# evaluate model
	score, history = evaluate_model(trainX, trainY, testX, testY)
	# learning curves
	summarize_diagnostics(history)
	# summarize estimated performance
	summarize_performance(score)
 
# entry point, run the test harness
main()