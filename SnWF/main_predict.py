from keras import backend as K
from keras.utils import np_utils
from keras.models import Model
from model.model_keras import *
import numpy as np
import time
import random
import os
import sklearn.metrics as metrics
from tqdm import tqdm
import tensorflow as tf
from util.data_loader import *
from keras.layers import Input
from util.keras_callbacks import *
from sklearn.model_selection import StratifiedShuffleSplit


models_filenames = ['weights/snWF-1.h5',
                    'weights/snWF-2.h5',
                    'weights/snWF-3.h5',
                    'weights/snWF-4.h5',
                    'weights/snWF-5.h5']

trainX, testX,trainY,testY = get_feature_dataloader()
INPUT_SHAPE = (505,1)
NB_CLASSES = 100
model = DFNet.build_BASE(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
best_acc = 0.0
best_weights = None

test_preds = []
for fn in models_filenames:
    model.load_weights(fn)
    print("Predicting test set values on model %s" % (fn))
    yPreds = model.predict(testX, batch_size=128, verbose=2)
    test_preds.append(yPreds)


weighted_predictions = np.zeros((testX.shape[0], NB_CLASSES), dtype='float32')
for prediction in test_preds:
    weighted_predictions += 0.2 * prediction
yPred = np.argmax(weighted_predictions, axis=1)
yTrue = np.argmax(testY, axis=1)
accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)
exit()


# if OPTIMIZE == 0:
#     with open('weights/Ensemble weights %s.json' % model_prefix, mode='r') as f:
#         dictionary = json.load(f)

#     prediction_weights = dictionary['best_weights']
#     calculate_weighted_accuracy()

# elif OPTIMIZE == -1:
#     prediction_weights = [1. / len(models_filenames)] * len(models_filenames)
#     calculate_weighted_accuracy()

# ''' OPTIMIZATION REGION '''

# def log_loss_func(weights):
#     ''' scipy minimize will pass the weights as a numpy array '''
#     final_prediction = np.zeros((trainX.shape[0], nb_classes), dtype='float32')

#     for weight, prediction in zip(weights, train_preds):
#         final_prediction += weight * prediction

#     return log_loss(trainY_cat, final_prediction)


# for iteration in range(NUM_TESTS):
#     prediction_weights = np.random.random(len(models_filenames))

#     constraints = ({'type': 'eq', 'fun':lambda w: 1 - sum(w)})
#     bounds = [(0, 1)] * len(train_preds)

#     result = minimize(log_loss_func, prediction_weights, method='SLSQP', bounds=bounds, constraints=constraints)

#     print('Best Ensemble Weights: {weights}'.format(weights=result['x']))

#     weights = result['x']
#     weighted_predictions = np.zeros((testX.shape[0], nb_classes), dtype='float32')

#     for weight, prediction in zip(weights, test_preds):
#         weighted_predictions += weight * prediction

#     yPred = np.argmax(weighted_predictions, axis=1)
#     yTrue = testY

#     accuracy = metrics.accuracy_score(yTrue, yPred) * 100
#     error = 100 - accuracy
#     print("Iteration %d: Accuracy : " % (iteration + 1), accuracy)
#     print("Iteration %d: Error : " % (iteration + 1), error)

#     if accuracy > best_acc:
#         best_acc = accuracy
#         best_weights = weights

#     print()

# print("Best Accuracy : ", best_acc)
# print("Best Weights : ", best_weights)

# with open('weights/Ensemble weights %s.json' % model_prefix, mode='w') as f:
#     dictionary = {'best_weights' : best_weights.tolist()}
#     json.dump(dictionary, f)