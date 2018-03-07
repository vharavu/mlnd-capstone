from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping


def make_model():
    """

    """

    model = Sequential()
    model.add(Dense(64, input_dim=3, kernel_initializer='RandomNormal', activation='relu'))
    model.add(Dropout(0.20))

    model.add(Dense(128, kernel_initializer='RandomNormal', activation='relu'))
    model.add(Dropout(0.30))

    model.add(Dense(64, kernel_initializer='RandomNormal', activation='relu'))
    model.add(Dropout(0.40))

    model.add(Dense(32, kernel_initializer='RandomNormal', activation='relu'))
    model.add(Dropout(0.20))

    model.add(Dense(1, kernel_initializer='RandomNormal'))
    return model


def make_wide_model():
    """

    """

    model = Sequential()
    model.add(Dense(64, input_dim=3, kernel_initializer='RandomNormal', activation='relu'))
    model.add(Dropout(0.20))

    model.add(Dense(256, kernel_initializer='RandomNormal', activation='relu'))
    model.add(Dropout(0.40))

    model.add(Dense(1, kernel_initializer='RandomNormal'))
    return model

#model3 = make_wide_model()
#print(model3.summary())
