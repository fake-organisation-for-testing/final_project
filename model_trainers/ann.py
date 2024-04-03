import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

def train_model(X_train, y_train):
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    ## initialise your neural network
    model = Sequential()
    ## set up your input layer
    model.add(Dense(units = 16, kernel_initializer = "uniform", activation = 'relu', input_dim = 3))
    # hidden layer one
    model.add(Dense(units = 8, kernel_initializer = "uniform", activation = 'relu'))
    # hidden layer two
    model.add(Dense(units = 4, kernel_initializer = "uniform", activation = 'relu'))
    # regularization
    model.add(Dropout(0.25))
    #output layer
    model.add(Dense(units = 1, kernel_initializer = "uniform", activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.20, callbacks=[early_stop])
    return model

