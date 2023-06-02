import csv
import pandas as pd
import os
import librosa
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt


def extract_features(file_name='', max_pad_len=50, mfcc_num=40):
    try:
        audio, sample_rate = librosa.load(
            file_name,
            res_type='kaiser_fast'
        )
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=mfcc_num)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs_padded = np.pad(mfccs,
                             pad_width=((0, 0),
                                        (0, pad_width)),
                             mode='constant')
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}\n{e}")
        return None
    # Return mfccs_padded if the sounds segments have a variable length
    return mfccs_padded


def make_metadata():
    """
    Reads the files in the chillanto folder and writes their names in a csv file
    :return: saves a CSV file to disk
    """
    path = '../chillanto'

    with open('chillanto_metadata.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_name', 'path', 'label', 'class_name'])
        for dirpath, name, filenames in os.walk(path):
            for filename in filenames:
                if filename.split('.')[1] == 'wav':
                    writer.writerow([filename,
                                     dirpath + '/' + filename,
                                     filename.split('.')[0],
                                     os.path.basename(dirpath)])


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        title = 'Normalized confusion matrix - ' + title
    else:
        title = 'Confusion matrix, without normalization - ' + title

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def load_chillanto(random_state, max_pad_len, mfcc_num):
    try:
        metadata = pd.read_csv('chillanto_metadata.csv')
    except:
        make_metadata()
        metadata = pd.read_csv('chillanto_metadata.csv')

    features = []
    for index, row in metadata.iterrows():
        file_name = os.path.join(os.path.abspath(row.path))
        class_label = row["class_name"]
        data = extract_features(file_name, max_pad_len, mfcc_num)
        features.append([data, class_label])

    features_df = pd.DataFrame(features, columns=['feature', 'class_label'])
    features_df.dropna(inplace=True)
    print(f'Finished feature extraction from {len(features_df)} files')

    x = np.array(features_df.feature.tolist())
    y = np.array(features_df.class_label.tolist())

    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        yy,
                                                        test_size=0.2,
                                                        random_state=random_state)

    return x_train, x_test, y_train, y_test, yy, features, features_df


if __name__ == '__main__':
    make_metadata()
