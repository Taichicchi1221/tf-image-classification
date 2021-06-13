from augmentation import RandomFlipLeftRight
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from augmentation import *

AUG_BATCH = 64


def display_items(ds, name, row=6, col=6):
    for i, (img, label) in enumerate(ds):
        plt.figure(figsize=(15, int(15*row/col)))
        for j in range(row*col):
            _label = label[j, ].numpy()
            _label = np.round(_label, 2)
            plt.subplot(row, col, j+1)
            plt.axis('off')
            plt.title(str(_label))
            plt.imshow(img[j, ].numpy())
        plt.savefig(
            f"/workspaces/tf-image-classification/augmentation/test_result/result_{i}_{name}.png"
        )
        if i == 2:
            break


def onehot(image, label):
    return image, tf.one_hot(label, 5)


def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, (224, 224, 3))

    return image, label


def transform(images, labels):
    r = RandomFlipLeftRight(0.5, AUG_BATCH, 224, 224, 3, 1221)
    images, labels = r.transform(images, labels)
    return images, labels


def main():
    ds, info = tfds.load("tf_flowers", split="train",
                         as_supervised=True, with_info=True)
    ds = ds.map(preprocess_image)
    ds = ds.map(onehot)
    ds = ds.batch(AUG_BATCH)
    transform_ds = ds.map(transform)
    display_items(ds, name="raw")
    display_items(transform_ds, name="transform")


if __name__ == "__main__":
    main()
