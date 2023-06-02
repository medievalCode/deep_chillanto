from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from datetime import datetime
from helpers.utilities import load_chillanto


def create_and_train_model():
    random_state = 2023
    max_pad_len = 44
    mfcc_num = 40
    x_train, x_test, y_train, y_test, yy, _, _ = load_chillanto(
        random_state=random_state,
        max_pad_len=max_pad_len,
        mfcc_num=mfcc_num,
    )

    num_rows = mfcc_num
    num_columns = max_pad_len
    num_channels = 1

    x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
    x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

    num_labels = yy.shape[1]

    # Construct model
    model = Sequential()
    model.add(
        Conv2D(
            filters=16,
            kernel_size=2,
            input_shape=(num_rows, num_columns, num_channels),
            activation="relu",
        )
    )
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=2, activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=2, activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=2, activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(num_labels, activation="softmax"))

    # Compile the model
    model.compile(
        loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam"
    )

    # Display model architecture summary
    model.summary()

    # Calculate pre-training accuracy
    score = model.evaluate(x_test, y_test, verbose=1)
    accuracy = 100 * score[1]

    print("Pre-training accuracy: %.4f%%" % accuracy)

    num_epochs = 72
    num_batch_size = 256

    checkpointer = ModelCheckpoint(
        filepath="saved_models/weights.best.basic_cnn.hdf5",
        verbose=1,
        save_best_only=True,
    )
    start = datetime.now()

    model.fit(
        x_train,
        y_train,
        batch_size=num_batch_size,
        epochs=num_epochs,
        validation_data=(x_test, y_test),
        callbacks=[checkpointer],
        verbose=1,
    )

    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    # Evaluating the model on the training and testing set
    score = model.evaluate(x_train, y_train, verbose=0)
    print(f"Training Accuracy: {score[1]}")

    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"Testing Accuracy: {score[1]}")

    model_json = model.to_json()
    with open("saved_models/model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("saved_models/model.h5")
    print("Saved model to disk")


if __name__ == "__main__":
    create_and_train_model()
