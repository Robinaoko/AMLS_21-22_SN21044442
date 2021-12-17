import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from tensorflow.keras import layers, models
from PIL import Image
from tqdm import tqdm
from glob import glob

def main():
    test_path = 'Data/test/image/'
    tumor_test = pd.read_csv('test_label.csv')
    train_path = 'Data/image/'
    tumor_label = pd.read_csv('label.csv')
    img_size = 224
    label = []
    nub_train = len(glob(train_path + '/*.jpg'))
    image_data = np.zeros((nub_train, img_size, img_size, 3), dtype=np.uint8)

    # import images
    i = 0
    for img_path in tqdm(glob(train_path + '/*.jpg')):
        img = Image.open(img_path)
        img = img.resize((img_size, img_size))
        arr = np.asarray(img)
        image_data[i, :, :, :] = arr
        i += 1

    for i in range(3000):
        data = tumor_label['label'][i]
        if data == 'no_tumor':
            label.append(0)
        else:
            label.append(1)

    test_label = []
    nub_test = len(glob(test_path + '/*.jpg'))
    test_data = np.zeros((nub_test, img_size, img_size, 3), dtype=np.uint8)

    i = 0
    for img_path in tqdm(glob(test_path + '/*.jpg')):
        img = Image.open(img_path)
        img = img.resize((img_size, img_size))
        arr = np.asarray(img)
        test_data[i, :, :, :] = arr
        i += 1

    for i in range(200):
        data = tumor_test['label'][i]
        if data == 'no_tumor':
            test_label.append(0)
        else:
            test_label.append(1)

    # scale processing
    image_data = np.array(image_data, dtype='float32') / 255.0
    label = np.array(label)
    print(image_data.shape)
    print(label.shape)

    # one-hot encoding
    lb = LabelBinarizer()
    label = lb.fit_transform(label)

    # scale processing
    x_test = np.array(test_data, dtype='float32') / 255.0
    test_label = np.array(test_label)

    # one-hot encoding
    lb = LabelBinarizer()
    y_test = lb.fit_transform(test_label)
    print(x_test.shape)

    # split the dataset
    x_train, x_val, y_train, y_val = train_test_split(image_data, label, test_size=0.2, random_state=3)

    # design the model
    model = models.Sequential([
        layers.Conv2D(filters=48, kernel_size=(11,11), strides=4, activation='relu', padding='same', input_shape=(224,224,3)),
        layers.MaxPooling2D(pool_size=(3,3),strides=2),
        layers.Conv2D(filters=128, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(3,3),strides=2),
        layers.Conv2D(filters=192, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
        layers.Conv2D(filters=192, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
        layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(3,3),strides=2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(2048, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(2048, activation='relu'),
        layers.Dense(1, activation="sigmoid")
    ])
    model.summary()

    # hyper parameter
    LR = 0.0001
    EPOCHS = 20
    BS = 32

    # train the model
    model.compile(optimizer=keras.optimizers.Adam(LR),
                  loss=keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                  epochs=EPOCHS, batch_size=BS)

    # plot the accuracy and loss
    N = np.arange(0, EPOCHS)
    plt.style.use("seaborn")
    plt.figure()
    plt.plot(N, history.history["loss"], label="train_loss")
    plt.plot(N, history.history["val_loss"], label="val_loss")
    plt.plot(N, history.history["accuracy"], label="train_acc")
    plt.plot(N, history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig('./output/alex_binary_plot.png')

    print("------testing------")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(test_acc)

    # save the model
    print("------saving the model------")
    model.save('./output/alex_binary.model')

if __name__ == '__main__':
    main()
