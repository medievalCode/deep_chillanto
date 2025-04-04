import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import model_from_json
from helpers.utilities import plot_confusion_matrix, load_chillanto

seed = 2023
np.random.seed(seed)
num_rows = 40
num_columns = 44
num_channels = 1

metadata = pd.read_csv('chillanto_metadata.csv')
labels = metadata.class_name.unique()
num_labels = len(labels)

# load json and create model
json_file = open("saved_models/model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models/model.h5")
print("Loaded model from disk")
loaded_model.compile(
    loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam"
)

print("Created model and loaded weights from file")

# load chillanto dataset
x_train, x_test, y_train, y_test, yy, _, features_df = load_chillanto(
    seed, max_pad_len=num_columns, mfcc_num=num_rows
)

x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)

score = loaded_model.evaluate(x_train, y_train, verbose=0)
print(f"Testing Accuracy: {score[1]}")

y_predict = loaded_model.predict(x_train, verbose=1)
ax = plot_confusion_matrix(
    y_train.argmax(axis=1),
    y_predict.argmax(axis=1),
    labels,
    normalize=True,
    title="Test Set",
    cmap=plt.cm.Blues,
)
plt.show()
