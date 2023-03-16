import tensorflow as tf
import keras.models
import keras.datasets.mnist
import keras.layers
import keras.losses
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import keras



mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#reshaping
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#Normalizing the pixel values of images
x_train = x_train.astype('float32')/255.0

x_test = x_test.astype('float32')/255.0

#Plotting images of dataset
# fig = plt.figure(figsize = (10,3))

# for i in range(20):

#     ax= fig.add_subplot(2, 10, i+1, xticks=[], yticks=[])

#     ax.imshow(np.squeeze(x_train[i]), cmap='gray')

#     ax.set_title(y_train[i])

# plt.show()

# shape = x_train.shape[1:]
# print(shape)



model = keras.models.Sequential(
    [
        keras.layers.Conv2D(32, input_shape=(28,28,1), kernel_size=(3, 3), strides=(3, 3), padding="same", activation="relu", name="conv2d_1"),
        keras.layers.MaxPooling2D(pool_size=(2,2), name="MaxPolling_1"),
        keras.layers.Conv2D(64, (3, 3), strides=(3, 3), padding="same", activation="relu", name="conv2d_2"),
        keras.layers.MaxPooling2D(pool_size=(2,2), name="maxPooling_2"),
        keras.layers.Flatten(name="flatten"),
        keras.layers.Dense(500, activation="relu", name="Dense"),        
        keras.layers.Dense(10, activation="softmax", name="Dense2")
        
    ]
)
print(model.summary())



# predictions = model(x_train[:1]).numpy()
# print("untrained predictions: {pred}".format(pred = predictions))

# probabilities = tf.nn.softmax(predictions).numpy()
# print("untrained probabilities: {prob}".format(prob = probabilities))

# # Lossfunction is introduced to return scalar loss
# loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# loss_fn(y_train[:1], predictions).numpy()


model.compile(optimizer="adam",
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])


model.fit(x_train, y_train, epochs=5)

evaluation = model.evaluate(x_test,y_test, verbose=2)
print("Evaluation: {eval}".format(eval = evaluation))

test_predictions = model.predict(x_test)
print(test_predictions)

train_predictions = model.predict(x_train)

y_prediction = np.argmax(test_predictions, axis=1)
y_train_predictions = np.argmax(train_predictions, axis=1)

# cm = confusion_matrix(y_test, y_prediction, normalize="pred")

# plt.imshow(cm, cmap=plt.cm.gray_r, interpolation="nearest")
# plt.show()


disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_prediction)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix: \n{disp.confusion_matrix}")

plt.show()

sample_weight = (y_prediction != y_test)
display_error_conf_matrix = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_prediction, sample_weight=sample_weight, normalize="true",values_format=".0%")
plt.show()

sample_weight = (y_train_predictions != y_train)
display_error_conf_matrix = metrics.ConfusionMatrixDisplay.from_predictions(y_train, y_train_predictions, sample_weight=sample_weight, normalize="true",values_format=".0%")
plt.show()