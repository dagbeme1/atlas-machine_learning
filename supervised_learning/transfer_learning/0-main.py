#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
preprocess_data = __import__('0-transfer').preprocess_data

K.learning_phase = K.backend.learning_phase
_, (X, Y) = K.datasets.cifar10.load_data()

# Preprocess the original images and convert labels to one-hot encoding
X_p, Y_p = preprocess_data(X, Y)

# Load the pre-trained model with input shape (32, 32, 3)
base_model = K.applications.DenseNet201(include_top=False,
                                        weights='imagenet',
                                        input_shape=(32, 32, 3),
                                        pooling='max',
                                        classes=1000)

# Extract features from the base model
features_train = base_model.predict(X_p)

# Create the dense head classifier
input_tensor = K.Input(shape=(features_train.shape[1],))
layer_256 = K.layers.Dense(units=256, activation='elu')
output = layer_256(input_tensor)
dropout = K.layers.Dropout(0.5)
output = dropout(output)
softmax = K.layers.Dense(units=10, activation='softmax')
output = softmax(output)
model = K.models.Model(inputs=input_tensor, outputs=output)

# Compile the model
model.compile(optimizer=K.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(features_train, Y_p, batch_size=32, epochs=30, verbose=1, shuffle=True)
