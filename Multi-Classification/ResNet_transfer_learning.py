from ResNet_model import resnet50
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import os
import glob
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer

def main():

    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94

    train_dir = 'Data/train'
    validation_dir = 'Data/val'
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    test_path = 'Data/test/image/'
    tumor_label = pd.read_csv('test_label.csv')
    img_size = 224
    label = []
    nub_train = len(glob.glob(test_path + '/*.jpg'))
    image_data = np.zeros((nub_train, img_size, img_size, 3), dtype=np.uint8)

    # import images
    i = 0
    for img_path in tqdm(glob.glob(test_path + '/*.jpg')):
        img = Image.open(img_path)
        img = img.resize((img_size, img_size))
        img = np.array(img).astype(np.float32)
        img = img - [_R_MEAN, _G_MEAN, _B_MEAN]
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

    im_height = 224
    im_width = 224
    batch_size = 16
    epochs = 100
    num_classes = 4

    # data pre-processing
    def pre_function(img):
        img = img - [_R_MEAN, _G_MEAN, _B_MEAN]
        return img

    # data generator with data augmentation
    train_image_generator = ImageDataGenerator(horizontal_flip=True,
                                               preprocessing_function=pre_function)

    validation_image_generator = ImageDataGenerator(preprocessing_function=pre_function)

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
                                                                  shuffle=False,
                                                                  target_size=(im_height, im_width),
                                                                  class_mode='categorical')
    total_val = val_data_gen.n
    print("using {} images for training, {} images for validation.".format(total_train,
                                                                           total_val))

    # design the model and use transfer learning
    base_model = resnet50(num_classes=4, include_top=False)
    pre_weights_path = './pretrain_weights/pretrain_weights.ckpt'
    assert len(glob.glob(pre_weights_path + "*")), "cannot find {}".format(pre_weights_path)
    base_model.load_weights(pre_weights_path)
    base_model.trainable = False
    base_model.summary()

    model = tf.keras.Sequential([base_model,
                                 tf.keras.layers.GlobalAvgPool2D(),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(1024, activation="relu"),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(num_classes),
                                 tf.keras.layers.Softmax()])
    model.summary()

    # train the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    history = model.fit(x=train_data_gen,
                        steps_per_epoch=total_train // batch_size,
                        epochs=epochs,
                        validation_data=val_data_gen,
                        validation_steps=total_val // batch_size)

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
    plt.savefig('./output/resnet_2_plot.png')

    # save the model
    print("------saving the model------")
    model.save('./output/resnet_2.model')


if __name__ == '__main__':
    main()
