import pandas as pd
from tensorflow.keras import layers
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import os
from PIL import Image
from tqdm import tqdm
from glob import glob
from sklearn.preprocessing import LabelBinarizer

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
nub_train = len(glob(test_path + '/*.jpg'))
image_data = np.zeros((nub_train, img_size, img_size, 3), dtype=np.uint8)

# import images
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

im_height = 224
im_width = 224
batch_size = 16
epochs = 20

# data generator with data augmentation
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

# design the Xception model
def entry_flow(inputs):
    x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    previous_block_activation = x # Set aside residual
    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for size in [128, 256, 728]:
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(size, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(size, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        residual = layers.Conv2D(size, 1, strides=2, padding='same')(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
    return x

def middle_flow(x, num_blocks=8):
    previous_block_activation = x
    for _ in range(num_blocks):
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(728, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(728, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(728, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.add([x, previous_block_activation])  # Add back residual
        previous_block_activation = x # Set aside next residual
    return x

def exit_flow(x, num_classes=4):
    previous_block_activation =x

    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(728, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(1024, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    residual = layers.Conv2D(1024,1,strides=2, padding='same' )(previous_block_activation)
    x = layers.add([x, residual])    #Add back residual)

    x = layers.SeparableConv2D(1536, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.SeparableConv2D(2048, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    return layers.Dense(num_classes, activation='softmax')(x)

inputs = keras.Input(shape=(224,224,3))
outputs =exit_flow(middle_flow(entry_flow(inputs)))
xception = keras.Model(inputs, outputs)
xception.summary()

optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.001)
xception.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/Xception.h5',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    monitor='val_loss')]
history = xception.fit(x=train_data_gen,
                        steps_per_epoch=total_train // batch_size,
                        epochs=epochs,
                        validation_data=val_data_gen,
                        validation_steps=total_val // batch_size,
                        callbacks=callbacks)

# plot
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
plt.savefig('./output/Xception.png')


test_loss, test_acc = xception.evaluate(x_test, y_test, verbose=1)
print(test_acc)

# save the model
print("------saving the model------")
xception.save('./output/Xception.model')
