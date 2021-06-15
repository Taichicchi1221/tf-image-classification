import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


class SingleAugmentationBase:
    def __init__(self, p: float = 1.0, seed: int = None):
        self.p = p
        self.seed = seed

    def do_transform(self, l):
        return tf.cast((tf.random.uniform([l], dtype=tf.float32, seed=self.seed) <= self.p), tf.float32)

    def transform(self, images, labels):
        batch_size = tf.shape(images)[0]
        do_transform = self.do_transform(batch_size)

        result_images = tf.TensorArray(dtype=tf.float32, size=batch_size)
        result_labels = tf.TensorArray(dtype=tf.float32, size=batch_size)

        for idx in tf.range(0, batch_size, dtype=tf.int32):
            tr_image, tr_label = self.single_transform(
                images[idx], labels[idx]
            )
            result_images = result_images.write(
                idx,
                (1 - do_transform[idx]) * images[idx] +
                do_transform[idx] * tr_image
            )
            result_labels = result_labels.write(
                idx,
                (1 - do_transform[idx]) * labels[idx] +
                do_transform[idx] * tr_label
            )

        return result_images.stack(), result_labels.stack()


class RandomFlipLeftRight(SingleAugmentationBase):
    def __init__(self, p: float = 1.0, seed: int = None):
        super().__init__(p=p, seed=seed)

    def single_transform(self, image, label):
        image = tf.image.flip_left_right(image)
        return image, label


class RandomFlipUpDown(SingleAugmentationBase):
    def __init__(self, p: float = 1.0, seed: int = None):
        super().__init__(p=p, seed=seed)

    def single_transform(self, image, label):
        image = tf.image.flip_up_down(image)
        return image, label


class RandomRotation(SingleAugmentationBase):
    def __init__(self, ang_range: float = np.pi / 4, fill_mode: str = "NEAREST", p: float = 1.0, seed: int = None):
        super().__init__(p=p, seed=seed)
        self.ang_range = ang_range
        self.fill_mode = fill_mode

    def single_transform(self, image, label):
        rotation_ang = tf.random.uniform(
            [], 0, self.ang_range, dtype=tf.float32, seed=self.seed
        )
        image = tfa.image.rotate(
            image,
            rotation_ang,
            fill_mode=self.fill_mode
        )
        return image, label


class RandomBrightness(SingleAugmentationBase):
    def __init__(self, max_delta: float = 0.5, p: float = 1.0, seed: int = None):
        super().__init__(p=p, seed=seed)
        self.max_delta = max_delta

    def single_transform(self, image, label):
        delta = tf.random.uniform(
            [], -self.max_delta, self.max_delta, dtype=tf.float32, seed=self.seed
        )
        image = tf.image.adjust_brightness(image, delta)
        return image, label


class RandomContrast(SingleAugmentationBase):
    def __init__(self, lower: float = 0.5, upper: float = 1.5, p: float = 1.0, seed: int = None):
        super().__init__(p=p, seed=seed)
        self.lower = lower
        self.upper = upper

    def single_transform(self, image, label):
        contrast_factor = tf.random.uniform(
            [], self.lower, self.upper, dtype=tf.float32, seed=self.seed
        )
        image = tf.image.adjust_contrast(image, contrast_factor)
        return image, label


class RandomHue(SingleAugmentationBase):
    def __init__(self, max_delta: float = 0.5, p: float = 1.0, seed: int = None):
        super().__init__(p=p, seed=seed)
        self.max_delta = max_delta

    def single_transform(self, image, label):
        delta = tf.random.uniform(
            [], -self.max_delta, self.max_delta, dtype=tf.float32, seed=self.seed
        )
        image = tf.image.adjust_hue(image, delta)
        return image, label


class RandomJpegQuality(SingleAugmentationBase):
    def __init__(self, min_jpeg_quality: int = 80, max_jpeg_quality: int = 100, p: float = 1.0, seed: int = None):
        super().__init__(p=p, seed=seed)
        self.min_jpeg_quality = min_jpeg_quality
        self.max_jpeg_quality = max_jpeg_quality

    def single_transform(self, image, label):
        jpeg_quality = tf.random.uniform(
            [], self.min_jpeg_quality, self.max_jpeg_quality, dtype=tf.int32, seed=self.seed
        )
        image = tf.image.adjust_jpeg_quality(image, jpeg_quality)
        return image, label


class RandomSaturation(SingleAugmentationBase):
    def __init__(self, lower: float = 0.5, upper: float = 1.5, p: float = 1.0, seed: int = None):
        super().__init__(p=p, seed=seed)
        self.lower = lower
        self.upper = upper

    def single_transform(self, image, label):
        saturation_factor = tf.random.uniform(
            [], self.lower, self.upper, dtype=tf.float32, seed=self.seed
        )
        image = tf.image.adjust_saturation(image, saturation_factor)
        return image, label
