import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block

def inception_network():
    """
    Function that builds an inception network as described in the paper:
    Going Deeper with Convolutions (2014)

    Returns:
        model (tensorflow.keras.Model): The built Inception Network model
    """
    # Use the He normal initializer for the weights
    initializer = K.initializers.he_normal()
    
    # Define the input tensor with shape (224, 224, 3)
    X = K.Input(shape=(224, 224, 3))

    # First Convolution Layer
    X = K.layers.Conv2D(filters=64,
                        kernel_size=7,
                        strides=2,
                        padding='same',
                        kernel_initializer=initializer,
                        activation='relu')(X)

    # MaxPooling Layer
    X = K.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(X)

    # Second Convolution Layer (1x1)
    X = K.layers.Conv2D(filters=64,
                        kernel_size=1,
                        strides=1,
                        padding='same',
                        kernel_initializer=initializer,
                        activation='relu')(X)

    # Third Convolution Layer (3x3)
    X = K.layers.Conv2D(filters=192,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        kernel_initializer=initializer,
                        activation='relu')(X)

    # MaxPooling Layer
    X = K.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(X)

    # Inception Blocks
    filters_list = [[64, 96, 128, 16, 32, 32],
                    [128, 128, 192, 32, 96, 64],
                    [192, 96, 208, 16, 48, 64],
                    [160, 112, 224, 24, 64, 64],
                    [128, 128, 256, 24, 64, 64],
                    [112, 144, 288, 32, 64, 64],
                    [256, 160, 320, 32, 128, 128],
                    [256, 160, 320, 32, 128, 128],
                    [384, 192, 384, 48, 128, 128]]

    for filters in filters_list:
        X = inception_block(X, filters)

    # Average Pooling Layer
    X = K.layers.AvgPool2D(pool_size=7, padding='same')(X)

    # Dropout Layer
    X = K.layers.Dropout(0.4)(X)

    # Fully Connected Layer with Softmax Activation
    X = K.layers.Dense(units=1000,
                       activation='softmax',
                       kernel_initializer=initializer,
                       kernel_regularizer=K.regularizers.l2())(X)

    # Instantiate a model from the Model class with input X and output X
    model = K.models.Model(inputs=X, outputs=X)

    return model
