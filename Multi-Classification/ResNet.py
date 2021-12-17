import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from ResNet_model import resnet50
from PIL import Image
from tqdm import tqdm
from glob import glob

def main():

    BS = 16
    epochs = 100
    num_classes = 4

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
        if data == 'glioma_tumor':
            label.append(0)
        elif data == 'meningioma_tumor':
            label.append(1)
        elif data == 'no_tumor':
            label.append(2)
        elif data == 'pituitary_tumor':
            label.append(3)

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
        if data == 'glioma_tumor':
            test_label.append(0)
        elif data == 'meningioma_tumor':
            test_label.append(1)
        elif data == 'no_tumor':
            test_label.append(2)
        elif data == 'pituitary_tumor':
            test_label.append(3)

    # scale processing
    image_data = np.array(image_data,dtype='float32')/255.0
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

    # split the data
    x_train, x_val, y_train, y_val = train_test_split(image_data, label, test_size=0.2, random_state=3)

    # import the model
    # base_model = resnet50(num_classes=4, include_top=False)
    base_model = keras.applications.resnet.ResNet50(weights="imagenet",
                                                  include_top=False)
    avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(4, activation="softmax")(avg)
    model = keras.Model(inputs=base_model.input, outputs=output)
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.000001),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=BS)

    N = np.arange(0, epochs)
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
    plt.savefig('./output/resnet50.png')

    print("------testing------")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(test_loss, test_acc)

    # save the model
    print("------saving the model------")
    model.save('./output/resnet50.model')


if __name__ == '__main__':
    main()