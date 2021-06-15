import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from single_augmentation import *
from mix_augmentation import *
from utils import *

AUG_BATCH = 16
H = 224
W = 224
CHANNELS = 3

tf.random.set_seed(1221)


def display_items(ds, name, row=4, col=4):
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
    image = tf.image.resize(image, (H, W))
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, (H, W, CHANNELS))

    return image, label


def transform(images, labels):
    tr = Compose(
        [
            # RandomFlipLeftRight(p=0.5),
            # RandomFlipUpDown(p=0.5),
            # RandomBrightness(p=0.5),
            # RandomContrast(p=0.5),
            # RandomHue(p=0.5),
            # RandomJpegQuality(p=0.5),
            # RandomRotation(p=0.5, fill_mode="REFLECT"),
            # RandomSaturation(p=0.5),
            # RandomMixUp(p=0.5),
            RandomCutMix(p=0.5),
        ]
    )

    images, labels = tr.transform(images, labels)

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
    print("#" * 30, f"process time: {time.process_time()}", "#" * 30)
