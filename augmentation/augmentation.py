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


class MixAugmentationBase:
    def __init__(self, p: float = 1.0, seed: int = None):
        self.p = p
        self.seed = seed

    def _beta_sampling(self, shape, alpha=1.0):
        r1 = tf.random.gamma(shape, alpha, 1, dtype=tf.float32)
        r2 = tf.random.gamma(shape, alpha, 1, dtype=tf.float32)
        return r1 / (r1 + r2)

    def transform(self, images, labels):
        return images, labels


class RandomMixUp(MixAugmentationBase):
    def __init__(self, alpha: float = 1.0, p: float = 1.0, seed: int = None):
        super().__init__(p=p, seed=seed)
        self.alpha = alpha

    def _calc_mixup(
        self,
        batch_size
    ):
        P = tf.cast(tf.random.uniform(
            [batch_size], 0, 1) <= self.p, tf.float32
        )
        a = self._beta_sampling([batch_size], self.alpha) * P
        return a, a

    def transform(
        self,
        images,
        labels,
    ):
        batch_size = tf.shape(images)[0]
        image_mix_ratios, label_mix_ratios = self._calc_mixup(batch_size)

        result_images = tf.TensorArray(dtype=tf.float32, size=batch_size)
        result_labels = tf.TensorArray(dtype=tf.float32, size=batch_size)

        for j in tf.range(0, batch_size, dtype=tf.int32):
            k = tf.random.uniform([], 0, batch_size, dtype=tf.int32)
            img1 = images[j]
            img2 = images[k]
            lab1 = labels[j]
            lab2 = labels[k]

            result_images = result_images.write(
                j,
                image_mix_ratios[j] * img1 + (1 - image_mix_ratios[j]) * img2
            )
            result_labels = result_labels.write(
                j,
                label_mix_ratios[j] * lab1 + (1 - label_mix_ratios[j]) * lab2
            )

        return result_images.stack(), result_labels.stack()


class RandomCutMix(MixAugmentationBase):
    def __init__(self, alpha: float = 1.0, p: float = 1.0, seed: int = None):
        super().__init__(p=p, seed=seed)
        self.alpha = alpha

    def _calc_cutmix(
        self,
        batch_size,
        image_size_0,
        image_size_1,
    ):
        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
        P = tf.cast(tf.random.uniform(
            [batch_size], 0, 1) <= self.p, tf.int32)
        # CHOOSE RANDOM LOCATION
        x = tf.random.uniform([batch_size], 0, image_size_1, dtype=tf.int32)
        y = tf.random.uniform([batch_size], 0, image_size_0, dtype=tf.int32)
        # this is beta dist with alpha=1.0
        b = self._beta_sampling([batch_size], self.alpha)
        WIDTH = tf.cast(
            tf.cast(image_size_1, dtype=tf.float32) *
            tf.math.sqrt(1 - b), tf.int32
        ) * P
        HEIGHT = tf.cast(
            tf.cast(image_size_0, dtype=tf.float32) *
            tf.math.sqrt(1 - b), tf.int32
        ) * P

        ya = tf.math.maximum(0, y - WIDTH//2)
        yb = tf.math.minimum(image_size_1, y + WIDTH // 2)
        xa = tf.math.maximum(0, x - WIDTH//2)
        xb = tf.math.minimum(image_size_0, x + WIDTH // 2)
        # MAKE CUTMIX RATIO
        image_ratios = tf.TensorArray(dtype=tf.float32, size=batch_size)
        label_ratios = tf.TensorArray(dtype=tf.float32, size=batch_size)
        for j in tf.range(0, batch_size, dtype=tf.int32):
            one = tf.ones(
                [yb[j, ] - ya[j, ], xa[j, ]], dtype=tf.float32
            )
            two = tf.zeros(
                [yb[j, ] - ya[j, ], xb[j, ] - xa[j, ]], dtype=tf.float32
            )
            three = tf.ones(
                [yb[j, ] - ya[j, ], image_size_1 - xb[j, ]], dtype=tf.float32
            )
            middle = tf.concat([one, two, three], axis=1)
            res_image_ratio = tf.concat(
                [
                    tf.ones(
                        [ya[j, ], image_size_1], dtype=tf.float32
                    ),
                    middle,
                    tf.ones(
                        [image_size_0 - yb[j, ], image_size_1], dtype=tf.float32
                    )
                ], axis=0
            )
            image_ratios = image_ratios.write(
                j, tf.expand_dims(res_image_ratio, -1)
            )

            # MAKE CUTMIX LABEL
            a = 1 - tf.cast(
                tf.cast(HEIGHT[j, ], tf.float32) *
                tf.cast(WIDTH[j, ], tf.float32) /
                tf.cast(image_size_0, tf.float32) /
                tf.cast(image_size_1, tf.float32),
                tf.float32
            )
            label_ratios = label_ratios.write(j, a)

        return image_ratios.stack(), label_ratios.stack()

    def transform(
        self,
        images,
        labels,
    ):
        batch_size = tf.shape(images)[0]
        image_size_0 = tf.shape(images)[1]
        image_size_1 = tf.shape(images)[2]

        image_mix_ratios, label_mix_ratios = self._calc_cutmix(
            batch_size,
            image_size_0,
            image_size_1,
        )

        result_images = tf.TensorArray(dtype=tf.float32, size=batch_size)
        result_labels = tf.TensorArray(dtype=tf.float32, size=batch_size)

        for j in tf.range(0, batch_size, dtype=tf.int32):
            k = tf.random.uniform([], 0, batch_size, dtype=tf.int32)

            result_images = result_images.write(
                j,
                image_mix_ratios[j] * images[j] +
                (1 - image_mix_ratios[j]) * images[k]
            )
            result_labels = result_labels.write(
                j,
                label_mix_ratios[j] * labels[j] +
                (1 - label_mix_ratios[j]) * labels[k]
            )

        return result_images.stack(), result_labels.stack()


class RandomFMix(MixAugmentationBase):
    def __init__(self, alpha: float = 1.0, decay: float = 3.0, p: float = 1.0, seed: int = None):
        super().__init__(p=p, seed=seed)
        self.alpha = alpha
        self.decay = decay

    def _fftfreq(self, n, d=1.0):
        # https://github.com/ecs-vlc/FMix/blob/master/fmix.py
        val = 1.0 / (n * d)
        N = (n - 1) // 2 + 1
        p1 = tf.range(0, N, dtype=tf.float32)
        p2 = tf.range(-(n // 2), 0, dtype=tf.float32)
        results = tf.concat([p1, p2], 0)
        return results * val

    # Parameters are 'h' and 'w' for simplicity.
    def _fftfreqnd(self, h, w):
        """ Get bin values for discrete fourier transform of size (h, w)
        """
        # In the original implementation, '[: w // 2 + 2]' or '[: w // 2 + 1]' is
        # applied to fx.  However, I don't do this here, because tf.signal.ifft2d
        # returns the same shape as the input.  tf.signal.ifft2d does not accept
        # shape, as in np.fft.irfftn used in the original code.  I think that
        # a tensor of width by height is necessary here for tf.signal.ifft2d.
        # [: w // 2 + 2] or [ : w // 2 + 1]
        fx = self._fftfreq(tf.cast(w, tf.float32))
        fy = self._fftfreq(tf.cast(h, tf.float32))

        fx_square = fx * fx
        fy_square = fy * fy
        return tf.math.sqrt(fx_square[tf.newaxis, :] + fy_square[:, tf.newaxis])

    def _get_spectrum(self, data_count, freqs, decay_power, IMAGE_MAX_W_H):
        # Make a tensor to scale frequencies, low frequencies are bigger
        # and high frequencies are smaller.
        # Make freqs greater than 0 to avoid division by 0.
        lowest_freq = 1. / IMAGE_MAX_W_H
        freqs_gt_zero = tf.math.maximum(freqs, lowest_freq)
        scale_hw = 1.0 / tf.math.pow(freqs_gt_zero, decay_power)

        # Generate random Gaussian distribution numbers of data_count x height x width x 2.
        # 2 in the last dimension is for real and imaginary part of a complex number.
        # In the original program, the first dimention is used for channels.
        # In this program, it is used for data in a batch.
        param_size = tf.concat([[data_count], tf.shape(
            freqs), [tf.constant(2, dtype=tf.int32)]], axis=0)
        param_bhw2 = tf.random.normal(param_size)

        # Make a spectrum by multiplying scale and param.  For scale,
        # expand first and last dimension for batch and real/imaginary part.
        scale_1hw1 = tf.expand_dims(scale_hw, -1)[tf.newaxis, :]
        spectrum_bhw2 = scale_1hw1 * param_bhw2
        return spectrum_bhw2

    def _make_low_freq_images(self, data_count, image_size_0, image_size_1, decay):
        IMAGE_MAX_W_H = tf.cast(tf.maximum(
            image_size_0, image_size_1), tf.float32)
        # Make a mask image by inverse Fourier transform of a spectrum,
        # which is generated by self._get_spectrum().
        freqs = self._fftfreqnd(image_size_0, image_size_1)
        spectrum_bhw2 = self._get_spectrum(
            data_count, freqs, decay, IMAGE_MAX_W_H)
        spectrum_re_bhw = spectrum_bhw2[:, :, :, 0]
        spectrum_im_bhw = spectrum_bhw2[:, :, :, 1]
        spectrum_comp_bhw = tf.complex(spectrum_re_bhw, spectrum_im_bhw)
        mask_bhw = tf.math.real(tf.signal.ifft2d(spectrum_comp_bhw))

        # Scale the mask values from 0 to 1.
        mask_min_b = tf.reduce_min(mask_bhw, axis=(-2, -1))
        mask_min_b11 = mask_min_b[:, tf.newaxis, tf.newaxis]
        mask_shift_to_0_bhw = mask_bhw - mask_min_b11
        mask_max_b = tf.reduce_max(mask_shift_to_0_bhw, axis=(-2, -1))
        mask_max_b11 = mask_max_b[:, tf.newaxis, tf.newaxis]
        mask_scaled_bhw = mask_shift_to_0_bhw / mask_max_b11
        return mask_scaled_bhw

    def _make_binary_masks(self, data_count, image_size_0, image_size_1, low_freq_images_bhw, mix_ratios_b):
        IMAGE_PIXEL_COUNT = image_size_0 * image_size_1
        # The goal is "top proportion of the image to have value ‘1’ and the rest to have value ‘0’".
        # To make this I use tf.scatter_nd().  For tf.scatter_nd(), indices and values
        # are necessary.

        # For each image, get indices of an image whose order is sorted from top to bottom.
        # These are used for row indices.  To combine with column indices, they are reshaped to 1D.
        low_freq_images_bp = tf.reshape(low_freq_images_bhw, [data_count, -1])
        row_indices_bp = tf.argsort(
            low_freq_images_bp, axis=-1, direction='DESCENDING', stable=True)
        row_indices_t = tf.reshape(row_indices_bp, [-1])

        # Make column indices, col_indices_t looks like
        # '[ 0 ... 0 1 ... 1 ..... data_count-1 ... data_count-1]'
        col_indices_b = tf.range(data_count, dtype=tf.int32)
        col_indices_t = tf.repeat(
            col_indices_b, IMAGE_PIXEL_COUNT, axis=-1)

        # Combine column and row indices for scatter_nd.
        scatter_indices_2t = tf.stack([col_indices_t, row_indices_t])
        scatter_indices_t2 = tf.transpose(scatter_indices_2t)

        # Make a tensor which looks like:
        # [[ 0.0 ... 1.0 ]   \  <-- tf.linspace(0.0, 1.0, self.IMAGE_PIXEL_COUNT)
        #   ...               | data_count
        #  [ 0.0 ... 1.0 ]]  /
        linspace_0_1_p = tf.linspace(0.0, 1.0, IMAGE_PIXEL_COUNT)
        linspace_0_1_1p = linspace_0_1_p[tf.newaxis, :]
        linspace_0_1_bp = tf.repeat(linspace_0_1_1p, data_count, axis=0)

        # Make mix_ratio of the top elements in each data '1' and the rest '0'
        # This looks like:
        # [[ 1.0 1.0 ... 0.0 ]   \    <-- top mix_ratios_b[0] elements are 1.0
        #   ...                   | data_count
        #  [ 1.0 1.0 ... 0.0 ]]  /    <-- top mix_ratios_b[data_count - 1] elements are 1.0
        mix_ratios_b1 = mix_ratios_b[:, tf.newaxis]
        scatter_updates_bp = tf.where(
            linspace_0_1_bp <= mix_ratios_b1, 1.0, 0.0)
        scatter_updates_t = tf.reshape(scatter_updates_bp, [-1])

        # Make binary masks by using tf.scatter_nd(), then reshape.
        bin_masks_bp = tf.scatter_nd(scatter_indices_t2, scatter_updates_t, [
            data_count, IMAGE_PIXEL_COUNT]
        )
        bin_masks_bhw1 = tf.reshape(
            bin_masks_bp, [data_count, image_size_0, image_size_1, 1]
        )
        return bin_masks_bhw1

    def _calc_fmix(
        self,
        batch_size,
        image_size_0,
        image_size_1,
    ):
        # Generate mix ratios by beta distribution.
        mix_ratios = self._beta_sampling(
            [batch_size], alpha=self.alpha)

        # Generate binary masks, then mix images.
        low_freq_images = self._make_low_freq_images(
            batch_size,
            image_size_0,
            image_size_1,
            self.decay
        )
        bin_masks = self._make_binary_masks(
            batch_size,
            image_size_0,
            image_size_1,
            low_freq_images,
            mix_ratios
        )

        return bin_masks, mix_ratios

    def transform(
        self,
        images,
        labels,
    ):
        batch_size = tf.shape(images)[0]
        image_size_0 = tf.shape(images)[1]
        image_size_1 = tf.shape(images)[2]

        image_mix_ratios, label_mix_ratios = self._calc_fmix(
            batch_size,
            image_size_0,
            image_size_1,
        )

        result_images = tf.TensorArray(dtype=tf.float32, size=batch_size)
        result_labels = tf.TensorArray(dtype=tf.float32, size=batch_size)

        for j in tf.range(0, batch_size, dtype=tf.int32):
            k = tf.random.uniform([], 0, batch_size, dtype=tf.int32)

            result_images = result_images.write(
                j,
                image_mix_ratios[j] * images[j] +
                (1 - image_mix_ratios[j]) * images[k]
            )
            result_labels = result_labels.write(
                j,
                label_mix_ratios[j] * labels[j] +
                (1 - label_mix_ratios[j]) * labels[k]
            )

        return result_images.stack(), result_labels.stack()


class Compose:
    def __init__(self, object_ls) -> None:
        self.object_ls = object_ls
        self.l = len(self.object_ls)

    def transform(self, images, labels):
        for obj in self.object_ls:
            images, labels = obj.transform(images, labels)

        return images, labels
