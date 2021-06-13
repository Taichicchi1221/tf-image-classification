import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from augmentation import *

AUG_BATCH = 64


def display_items(ds, name, row=5, col=5):
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
        if i == 5:
            break


def onehot(image, label):
    return image, tf.one_hot(label, 5)


def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, (224, 224, 3))

    return image, label


def transform(images, labels):
    transform_objects = [
        # RandomFlipLeftRight(0.5, AUG_BATCH, 224, 224, 3, 42),
        # RandomFlipUpDown(0.5, AUG_BATCH, 224, 224, 3, 43),
        # RandomRotation(0.5, AUG_BATCH, 224, 224, 3, 44,
        #                ang_range=np.pi, fill_mode="nearest"),
        # RandomBrightness(0.5, AUG_BATCH, 224, 224, 3, 45, max_delta=0.5),
        # RandomContrast(0.5, AUG_BATCH, 224, 224, 3,
        #                46, lower=0.5, upper=1.5),
        # RandomHue(0.5, AUG_BATCH, 224, 224, 3,
        #           47, max_delta=1.3),
        # RandomJpegQuality(0.5, AUG_BATCH, 224, 224, 3,
        #                   48, min_jpeg_quality=80, max_jpeg_quality=100),
        # RandomSaturation(0.5, AUG_BATCH, 224, 224, 3,
        #                  49, lower=0.5, upper=1.5),
        # RandomMixUp(0.5, 64, 224, 224, 3, 10001, 1.0),
        RandomCutMix(0.5, 64, 224, 224, 3, 10001, 1.0),
        # RandomFMix(0.5, 64, 224, 224, 3, 10001, 1.0, 3.0),
    ]

    for o in transform_objects:
        images, labels = o.transform(images, labels)

    return images, labels


def main():
    ds, info = tfds.load(
        "tf_flowers",
        split="train",
        as_supervised=True,
        with_info=True
    )
    ds = ds.map(preprocess_image)
    ds = ds.map(onehot)
    ds = ds.batch(AUG_BATCH)

    display_items(ds, name="raw")

    tr_ds, info = tfds.load(
        "tf_flowers",
        split="train",
        as_supervised=True,
        with_info=True
    )
    tr_ds = tr_ds.map(preprocess_image)
    tr_ds = tr_ds.map(onehot)
    tr_ds = tr_ds.batch(AUG_BATCH)
    tr_ds = tr_ds.map(transform)
    display_items(tr_ds, name="transform")


if __name__ == "__main__":
    main()
