#!/usr/bin/env python3

"""
Transfer Learning

The following script can be run on Google Colab
with the following libraries:

!python3 --version
Python 3.6.9
print(tf.__version__)
2.2.0
print(K.__version__)
2.3.0-tf
print(np.__version__)
1.18.5
print(matplotlib.__version__)
3.2.2

Alternative library versioning may incur errors
due to compatibility issues between deprecated
libraries on Google Colab
"""

import tensorflow as tf
import tensorflow.keras as K
import numpy as np

def preprocess_data(X, Y):
    """function that pre-processes the data"""
    X = K.applications.densenet.preprocess_input(X)
    # Using Keras' densenet.preprocess_input method to preprocess the input data 'X'.
    Y = K.utils.to_categorical(Y)
    # Using Keras' to_categorical method to convert the target labels 'Y' to one-hot encodings.
    return X, Y
    # Returning the preprocessed data 'X' and one-hot encoded labels 'Y'.


if __name__ == '__main__':
    # This block of code is executed only when the script is run directly, not when it is imported as a module.

    # load the Cifar10 dataset, 50,000 training images and 10,000 test images
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    # Loading the CIFAR-10 dataset using Keras' load_data function. The data is split into training and test sets.

    # preprocess the data using the application's preprocess_input method
    # and convert the labels to one-hot encodings
    x_train, y_train = preprocess_data(x_train, y_train)
    # Preprocessing the training data using the preprocess_data function defined earlier.
    x_test, y_test = preprocess_data(x_test, y_test)
    # Preprocessing the test data using the preprocess_data function defined earlier.

    # instantiate a pre-trained model from the Keras API
    input_tensor = K.Input(shape=(32, 32, 3))
    # Creating an input tensor with shape (32, 32, 3) using Keras' Input function. This defines the shape of the input images for the model.

    # upsampling helps improve the validation accuracy to some extent
    # (insufficient here):
    # output = K.layers.UpSampling2D(size=(2, 2),
    #                                interpolation='nearest')(input_tensor)
    # The above code uses upsampling to improve validation accuracy, but it's commented out in the current script.

    # another approach: resize images to the image size upon which the network
    # was pre-trained:
    resized_images = K.layers.Lambda(
        lambda image: tf.image.resize(image, (224, 224)))(input_tensor)
    # Using Lambda layer to resize the images from (32, 32, 3) to (224, 224, 3).

    base_model = K.applications.DenseNet201(include_top=False,
                                            weights='imagenet',
                                            input_tensor=resized_images,
                                            input_shape=(224, 224, 3),
                                            pooling='max',
                                            classes=1000)
    # Creating a DenseNet201 base model, excluding the top classification layers. The model is pre-trained on ImageNet.

    output = base_model.layers[-1].output
    # Getting the output tensor from the last layer of the base model.

    base_model = K.models.Model(inputs=input_tensor, outputs=output)
    # Creating a Keras Model using the input tensor 'input_tensor' and the output tensor 'output'.

    # extract the bottleneck features (output feature maps)
    # from the pre-trained network (here, base-model)
    train_datagen = K.preprocessing.image.ImageDataGenerator()
    # Creating an ImageDataGenerator for the training data.

    train_generator = train_datagen.flow(x_train,
                                         y_train,
                                         batch_size=32,
                                         shuffle=False)
    # Creating a data generator for the training data.

    features_train = base_model.predict(train_generator)
    # Extracting bottleneck features (output feature maps) from the base model for the training data.

    # repeat the same operation with the test data (here used for validation)
    val_datagen = K.preprocessing.image.ImageDataGenerator()
    # Creating an ImageDataGenerator for the validation data (test data).

    val_generator = val_datagen.flow(x_test,
                                     y_test,
                                     batch_size=32,
                                     shuffle=False)
    # Creating a data generator for the validation data.

    features_valid = base_model.predict(val_generator)
    # Extracting bottleneck features (output feature maps) from the base model for the validation data (test data).

    # create a densely-connected head classifier
    initializer = K.initializers.he_normal()
    # Using He normal initializer to initialize the weights of the dense layers.

    input_tensor = K.Input(shape=features_train.shape[1])
    # Creating an input tensor with shape (num_features,) where num_features is the number of features extracted by the base model.

    layer_256 = K.layers.Dense(units=256,
                               activation='elu',
                               kernel_initializer=initializer,
                               kernel_regularizer=K.regularizers.l2())
    # Adding a dense (fully connected) layer with 256 units, 'elu' activation, He normal initializer, and L2 regularization.

    output = layer_256(input_tensor)
    # Connecting the input tensor to the layer_256.

    dropout = K.layers.Dropout(0.5)
    # Adding a dropout layer with a dropout rate of 0.5 to prevent overfitting.

    output = dropout(output)
    # Connecting the output of layer_256 to the dropout layer.

    softmax = K.layers.Dense(units=10,
                             activation='softmax',
                             kernel_initializer=initializer,
                             kernel_regularizer=K.regularizers.l2())
    # Adding the final dense (fully connected) layer with 10 units (for 10 classes), 'softmax' activation, He normal initializer, and L2 regularization.

    output = softmax(output)
    # Connecting the output of the dropout layer to the final softmax layer.

    model = K.models.Model(inputs=input_tensor, outputs=output)
    # Creating a Keras Model using the input tensor 'input_tensor' and the output tensor 'output'.

    # compile the densely-connected head classifier
    model.compile(optimizer=K.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Compiling the model with Adam optimizer, categorical cross-entropy loss (for multiclass classification), and accuracy as the evaluation metric.

    # reduce learning rate when val_accuracy has stopped improving
    lr_reduce = K.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                              factor=0.6,
                                              patience=2,
                                              verbose=1,
                                              mode='max',
                                              min_lr=1e-7)
    # Adding a learning rate reduction callback to reduce learning rate when validation accuracy stops improving.

    # stop training when val_accuracy has stopped improving
    early_stop = K.callbacks.EarlyStopping(monitor='val_accuracy',
                                           patience=3,
                                           verbose=1,
                                           mode='max')
    # Adding an early stopping callback to stop training when validation accuracy stops improving.

    # callback to save the Keras model and (best) weights obtained
    # on an epoch basis
    checkpoint = K.callbacks.ModelCheckpoint('cifar10.h5',
                                             monitor='val_accuracy',
                                             verbose=1,
                                             save_weights_only=False,
                                             save_best_only=True,
                                             mode='max',
                                             save_freq='epoch')
    # Adding a callback to save the model and weights on the basis of validation accuracy.

    # train the densely-connected head classifier
    history = model.fit(features_train, y_train,
                        batch_size=32,
                        epochs=20,
                        verbose=1,
                        callbacks=[lr_reduce, early_stop, checkpoint],
                        validation_data=(features_valid, y_test),
                        shuffle=True)
    # Training the model using the extracted features from the base model. The training data is augmented using the ImageDataGenerator.
