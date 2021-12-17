import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import json
import os
from PIL import Image
from tqdm import tqdm
from glob import glob
from sklearn.preprocessing import LabelBinarizer

def main():

    # import images
    train_dir = 'Data/train'
    validation_dir = 'Data/val'
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    test_path = 'Data/test/image/'
    tumor_label = pd.read_csv('test_label.csv')
    img_size = 227
    label = []
    nub_train = len(glob(test_path + '/*.jpg'))
    image_data = np.zeros((nub_train, img_size, img_size, 3), dtype=np.uint8)

    i = 0
    for img_path in tqdm(glob(test_path + '/*.jpg')):
        img = Image.open(img_path)
        img = img.resize((img_size, img_size))
        arr = np.asarray(img)
        image_data[i, :, :, :] = arr
        i += 1

    for i in range(200):
        data = tumor_label['label'][i]
        if data == 'glioma_tumor':
            label.append(0)
        elif data == 'meningioma_tumor':
            label.append(1)
        elif data == 'no_tumor':
            label.append(2)
        elif data == 'pituitary_tumor':
            label.append(3)

    # scale processing
    x_test = np.array(image_data, dtype='float32') / 255.0
    label = np.array(label)

    # one-hot encoding
    lb = LabelBinarizer()
    y_test = lb.fit_transform(label)
    print(x_test.shape)

    im_height = 227
    im_width = 227
    batch_size = 32
    epochs = 100

    # horizontal flip and import datas
    train_image_generator = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)
    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(im_height, im_width),
                                                               class_mode='categorical')
    total_train = train_data_gen.n

    # get class dict
    class_indices = train_data_gen.class_indices
    inverse_dict = dict((val, key) for key, val in class_indices.items())

    json_str = json.dumps(inverse_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                  batch_size=batch_size,
                                                                  shuffle=True,
                                                                  target_size=(im_height, im_width),
                                                                  class_mode='categorical')
    total_val = val_data_gen.n
    print("using {} images for training, {} images for validation.".format(total_train,
                                                                           total_val))

    # design the model
    model = models.Sequential([
        layers.Conv2D(filters=48, kernel_size=(11, 11), strides=4, activation='relu', padding='same',
                      input_shape=(227, 227, 3)),
        layers.MaxPooling2D(pool_size=(3, 3), strides=2),
        layers.Conv2D(filters=128, kernel_size=(5, 5), strides=1, activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(3, 3), strides=2),
        layers.Conv2D(filters=192, kernel_size=(3, 3), strides=1, activation='relu', padding='same'),
        layers.Conv2D(filters=192, kernel_size=(3, 3), strides=1, activation='relu', padding='same'),
        layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(3, 3), strides=2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(2048, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(2048, activation='relu'),
        layers.Dense(4, activation="softmax")
    ])
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/Alexnet.h5',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    monitor='val_loss')]

    history = model.fit(x=train_data_gen,
                        steps_per_epoch=total_train // batch_size,
                        epochs=epochs,
                        validation_data=val_data_gen,
                        validation_steps=total_val // batch_size,
                        callbacks=callbacks)

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
    plt.savefig('./output/Alexnet_plot.png')

    print("------testing------")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(test_loss, test_acc)

    # save the model
    print("------saving the model------")
    model.save('./output/Alexnet.model')


if __name__ == '__main__':
    main()
