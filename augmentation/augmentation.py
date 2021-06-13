from dataclasses import dataclass
import tensorflow as tf
import tensorflow_addons as tfa


class AugmentationBase:
    def __init__(self, p, aug_batch_size, image_size_0, image_size_1, image_channels, seed):
        self.p = p
        self.aug_batch_size = aug_batch_size
        self.image_size_0 = image_size_0
        self.image_size_1 = image_size_1
        self.image_channels = image_channels
        self.seed = seed

    def do_transform(self):
        return tf.cast((tf.random.uniform([self.aug_batch_size], dtype=tf.float32, seed=self.seed) <= self.p), tf.float32)

    def transform(self, images, labels):
        imgs = []
        labs = []
        a = self.do_transform()
        for idx in range(self.aug_batch_size):
            tr_image, tr_label = self.single_transform(
                images[idx], labels[idx]
            )
            imgs.append((1 - a[idx]) * images[idx] + a[idx] * tr_image)
            labs.append((1 - a[idx]) * labels[idx] + a[idx] * tr_label)

        result_images = tf.reshape(
            tf.stack(imgs),
            (
                self.aug_batch_size,
                self.image_size_0,
                self.image_size_1,
                self.image_channels
            )
        )
        result_labels = tf.reshape(
            tf.stack(labs),
            (self.aug_batch_size, -1)
        )
        return result_images, result_labels


class RandomFlipLeftRight(AugmentationBase):
    def __init__(self, p, aug_batch_size, image_size_0, image_size_1, image_channels, seed):
        super().__init__(p, aug_batch_size, image_size_0,
                         image_size_1, image_channels, seed)

    def single_transform(self, image, label):
        image = tf.image.flip_left_right(image)
        return image, label


class RandomFlipUpDown(AugmentationBase):
    def __init__(self, p, aug_batch_size, image_size_0, image_size_1, image_channels, seed):
        super().__init__(p, aug_batch_size, image_size_0,
                         image_size_1, image_channels, seed)

    def single_transform(self, image, label):
        image = tf.image.flip_up_down(image)
        return image, label


class RandomRotation(AugmentationBase):
    def __init__(self, p, aug_batch_size, image_size_0, image_size_1, image_channels, seed, ang_range, fill_mode):
        super().__init__(p, aug_batch_size, image_size_0,
                         image_size_1, image_channels, seed)
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


class RandomBrightness(AugmentationBase):
    def __init__(self, p, aug_batch_size, image_size_0, image_size_1, image_channels, seed, max_delta):
        super().__init__(p, aug_batch_size, image_size_0,
                         image_size_1, image_channels, seed)
        self.max_delta = max_delta

    def single_transform(self, image, label):
        delta = tf.random.uniform(
            [], -self.max_delta, self.max_delta, dtype=tf.float32, seed=self.seed
        )
        image = tf.image.adjust_brightness(image, delta)
        return image, label


class RandomContrast(AugmentationBase):
    def __init__(self, p, aug_batch_size, image_size_0, image_size_1, image_channels, seed, lower, upper):
        super().__init__(p, aug_batch_size, image_size_0,
                         image_size_1, image_channels, seed)
        self.lower = lower
        self.upper = upper

    def single_transform(self, image, label):
        contrast_factor = tf.random.uniform(
            [], self.lower, self.upper, dtype=tf.float32, seed=self.seed
        )
        image = tf.image.adjust_contrast(image, contrast_factor)
        return image, label


class RandomHue(AugmentationBase):
    def __init__(self, p, aug_batch_size, image_size_0, image_size_1, image_channels, seed, max_delta):
        super().__init__(p, aug_batch_size, image_size_0,
                         image_size_1, image_channels, seed)
        self.max_delta = max_delta

    def single_transform(self, image, label):
        delta = tf.random.uniform(
            [], -self.max_delta, self.max_delta, dtype=tf.float32, seed=self.seed
        )
        image = tf.image.adjust_hue(image, delta)
        return image, label


class RandomJpegQuality(AugmentationBase):
    def __init__(self, p, aug_batch_size, image_size_0, image_size_1, image_channels, seed, min_jpeg_quality, max_jpeg_quality):
        super().__init__(p, aug_batch_size, image_size_0,
                         image_size_1, image_channels, seed)
        self.min_jpeg_quality = min_jpeg_quality
        self.max_jpeg_quality = max_jpeg_quality

    def single_transform(self, image, label):
        jpeg_quality = tf.random.uniform(
            [], self.min_jpeg_quality, self.max_jpeg_quality, dtype=tf.int32, seed=self.seed
        )
        image = tf.image.adjust_jpeg_quality(image, jpeg_quality)
        return image, label


class RandomSaturation(AugmentationBase):
    def __init__(self, p, aug_batch_size, image_size_0, image_size_1, image_channels, seed, lower, upper):
        super().__init__(p, aug_batch_size, image_size_0,
                         image_size_1, image_channels, seed)
        self.lower = lower
        self.upper = upper

    def single_transform(self, image, label):
        saturation_factor = tf.random.uniform(
            [], self.lower, self.upper, dtype=tf.float32, seed=self.seed
        )
        image = tf.image.adjust_saturation(image, saturation_factor)
        return image, label


class MixAugmentationBase:
    def __init__(self, p, aug_batch_size, image_size_0, image_size_1, image_channels, seed):
        self.p = p
        self.aug_batch_size = aug_batch_size
        self.image_size_0 = image_size_0
        self.image_size_1 = image_size_1
        self.image_channels = image_channels
        self.seed = seed

    def _beta_sampling(self, shape, alpha=1.0):
        r1 = tf.random.gamma(shape, alpha, 1, dtype=tf.float32)
        r2 = tf.random.gamma(shape, alpha, 1, dtype=tf.float32)
        return r1 / (r1 + r2)

    def transform(self, images, labels):
        return images, labels


class RandomMixUp(MixAugmentationBase):
    def __init__(self, p, aug_batch_size, image_size_0, image_size_1, image_channels, seed, alpha):
        super().__init__(p, aug_batch_size,
                         image_size_0, image_size_1, image_channels, seed)
        self.alpha = alpha

    def _calc_mixup(
        self,
    ):
        P = tf.cast(tf.random.uniform(
            [self.aug_batch_size], 0, 1) <= self.p, tf.float32)
        a = self._beta_sampling([self.aug_batch_size], self.alpha) * P
        return a, a

    def transform(
        self,
        image_batch,
        label_batch,
    ):
        imgs = []
        labs = []
        image_mix_ratios, label_mix_ratios = self._calc_mixup()

        for j in range(self.aug_batch_size):
            k = tf.cast(tf.random.uniform(
                [], 0, self.aug_batch_size), tf.int32)
            img1 = image_batch[j, ]
            img2 = image_batch[k, ]
            lab1 = label_batch[j, ]
            lab2 = label_batch[k, ]

            result_image = image_mix_ratios[j, ] * \
                img1 + (1 - image_mix_ratios[j, ]) * img2
            imgs.append(result_image)
            result_label = label_mix_ratios[j, ] * \
                lab1 + (1 - label_mix_ratios[j, ]) * lab2
            labs.append(result_label)

        result_image_batch = tf.reshape(tf.stack(
            imgs), (self.aug_batch_size, self.image_size_0, self.image_size_1, self.image_channels))
        result_label_batch = tf.reshape(
            tf.stack(labs), (self.aug_batch_size, -1))

        return result_image_batch, result_label_batch


class RandomCutMix(MixAugmentationBase):
    def __init__(self, p, aug_batch_size, image_size_0, image_size_1, image_channels, seed, alpha):
        super().__init__(p, aug_batch_size,
                         image_size_0, image_size_1, image_channels, seed)
        self.alpha = alpha

    def _calc_cutmix(
        self,
    ):
        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
        P = tf.cast(tf.random.uniform(
            [self.aug_batch_size], 0, 1) <= self.p, tf.int32)
        # CHOOSE RANDOM LOCATION
        x = tf.cast(tf.random.uniform(
            [self.aug_batch_size], 0, self.image_size_1), tf.int32)
        y = tf.cast(tf.random.uniform(
            [self.aug_batch_size], 0, self.image_size_0), tf.int32)
        # this is beta dist with alpha=1.0
        b = self._beta_sampling([self.aug_batch_size], self.alpha)
        WIDTH = tf.cast(self.image_size_1 * tf.math.sqrt(1 - b), tf.int32) * P
        HEIGHT = tf.cast(self.image_size_0 * tf.math.sqrt(1 - b), tf.int32) * P

        ya = tf.math.maximum(0, y - WIDTH//2)
        yb = tf.math.minimum(self.image_size_1, y + WIDTH // 2)
        xa = tf.math.maximum(0, x - WIDTH//2)
        xb = tf.math.minimum(self.image_size_0, x + WIDTH // 2)
        # MAKE CUTMIX RATIO
        image_ratios = []
        label_ratios = []
        for j in range(self.aug_batch_size):
            one = tf.ones([yb[j, ] - ya[j, ], xa[j, ]], dtype=tf.float32)
            two = tf.zeros([yb[j, ] - ya[j, ], xb[j, ] -
                            xa[j, ]], dtype=tf.float32)
            three = tf.ones(
                [yb[j, ] - ya[j, ], self.image_size_1 - xb[j, ]], dtype=tf.float32)
            middle = tf.concat([one, two, three], axis=1)
            res_image_ratio = tf.concat(
                [
                    tf.ones([ya[j, ], self.image_size_1], dtype=tf.float32),
                    middle,
                    tf.ones([self.image_size_0 - yb[j, ],
                             self.image_size_1], dtype=tf.float32)
                ], axis=0
            )
            image_ratios.append(tf.expand_dims(res_image_ratio, -1))

            # MAKE CUTMIX LABEL
            a = 1 - tf.cast(HEIGHT[j, ] * WIDTH[j, ] /
                            self.image_size_0 / self.image_size_1, tf.float32)
            label_ratios.append(a)

        return tf.stack(image_ratios), tf.stack(label_ratios)

    def transform(
        self,
        images,
        labels,
    ):
        imgs = []
        labs = []
        image_mix_ratios, label_mix_ratios = self._calc_cutmix()
        for j in range(self.aug_batch_size):
            k = tf.cast(tf.random.uniform(
                [], 0, self.aug_batch_size), tf.int32)
            img1 = images[j, ]
            img2 = images[k, ]
            lab1 = labels[j, ]
            lab2 = labels[k, ]

            result_image = image_mix_ratios[j, ] * \
                img1 + (1 - image_mix_ratios[j, ]) * img2
            imgs.append(result_image)
            result_label = label_mix_ratios[j, ] * \
                lab1 + (1 - label_mix_ratios[j, ]) * lab2
            labs.append(result_label)

        result_image_batch = tf.reshape(tf.stack(
            imgs), (self.aug_batch_size, self.image_size_0, self.image_size_1, self.image_channels))
        result_label_batch = tf.reshape(
            tf.stack(labs), (self.aug_batch_size, -1))

        return result_image_batch, result_label_batch


class RandomFMix(MixAugmentationBase):
    def __init__(self, p, aug_batch_size, image_size_0, image_size_1, image_channels, seed, alpha, decay):
        super().__init__(p, aug_batch_size,
                         image_size_0, image_size_1, image_channels, seed)
        self.alpha = alpha
        self.decay = decay
        self.IMAGE_MAX_W_H = max(self.image_size_0, self.image_size_1)
        self.IMAGE_PIXEL_COUNT = self.image_size_0 * self.image_size_1

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
        fx = self._fftfreq(w)  # [: w // 2 + 2] or [ : w // 2 + 1]
        fy = self._fftfreq(h)

        fx_square = fx * fx
        fy_square = fy * fy
        return tf.math.sqrt(fx_square[tf.newaxis, :] + fy_square[:, tf.newaxis])

    def _get_spectrum(self, data_count, freqs, decay_power):
        # Make a tensor to scale frequencies, low frequencies are bigger
        # and high frequencies are smaller.
        # Make freqs greater than 0 to avoid division by 0.
        lowest_freq = tf.constant(1. / self.IMAGE_MAX_W_H)
        freqs_gt_zero = tf.math.maximum(freqs, lowest_freq)
        scale_hw = 1.0 / tf.math.pow(freqs_gt_zero, decay_power)

        # Generate random Gaussian distribution numbers of data_count x height x width x 2.
        # 2 in the last dimension is for real and imaginary part of a complex number.
        # In the original program, the first dimention is used for channels.
        # In this program, it is used for data in a batch.
        param_size = [data_count] + list(freqs.shape) + [2]
        param_bhw2 = tf.random.normal(param_size)

        # Make a spectrum by multiplying scale and param.  For scale,
        # expand first and last dimension for batch and real/imaginary part.
        scale_1hw1 = tf.expand_dims(scale_hw, -1)[tf.newaxis, :]
        spectrum_bhw2 = scale_1hw1 * param_bhw2
        return spectrum_bhw2

    def _make_low_freq_images(self, data_count, decay):
        # Make a mask image by inverse Fourier transform of a spectrum,
        # which is generated by self._get_spectrum().
        freqs = self._fftfreqnd(self.image_size_0, self.image_size_1)
        spectrum_bhw2 = self._get_spectrum(data_count, freqs, decay)
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

    def _make_binary_masks(self, data_count, low_freq_images_bhw, mix_ratios_b):
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
            col_indices_b, self.IMAGE_PIXEL_COUNT, axis=-1)

        # Combine column and row indices for scatter_nd.
        scatter_indices_2t = tf.stack([col_indices_t, row_indices_t])
        scatter_indices_t2 = tf.transpose(scatter_indices_2t)

        # Make a tensor which looks like:
        # [[ 0.0 ... 1.0 ]   \  <-- tf.linspace(0.0, 1.0, self.IMAGE_PIXEL_COUNT)
        #   ...               | data_count
        #  [ 0.0 ... 1.0 ]]  /
        linspace_0_1_p = tf.linspace(0.0, 1.0, self.IMAGE_PIXEL_COUNT)
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
            data_count, self.IMAGE_PIXEL_COUNT])
        bin_masks_bhw1 = tf.reshape(
            bin_masks_bp, [data_count, self.image_size_0, self.image_size_1, 1])
        return bin_masks_bhw1

    def _calc_fmix(
        self,
    ):
        # Generate mix ratios by beta distribution.
        mix_ratios = self._beta_sampling(
            [self.aug_batch_size], alpha=self.alpha)

        # Generate binary masks, then mix images.
        low_freq_images = self._make_low_freq_images(
            self.aug_batch_size, self.decay)
        bin_masks = self._make_binary_masks(
            self.aug_batch_size, low_freq_images, mix_ratios)

        return bin_masks, mix_ratios

    def transform(
        self,
        image_batch,
        label_batch,
    ):
        imgs = []
        labs = []
        image_mix_ratios, label_mix_ratios = self._calc_fmix()
        for j in range(self.aug_batch_size):
            k = tf.cast(tf.random.uniform(
                [], 0, self.aug_batch_size), tf.int32)
            img1 = image_batch[j, ]
            img2 = image_batch[k, ]
            lab1 = label_batch[j, ]
            lab2 = label_batch[k, ]

            result_image = image_mix_ratios[j, ] * \
                img1 + (1 - image_mix_ratios[j, ]) * img2
            imgs.append(result_image)
            result_label = label_mix_ratios[j, ] * \
                lab1 + (1 - label_mix_ratios[j, ]) * lab2
            labs.append(result_label)

        result_image_batch = tf.reshape(tf.stack(
            imgs), (self.aug_batch_size, self.image_size_0, self.image_size_1, self.image_channels))
        result_label_batch = tf.reshape(
            tf.stack(labs), (self.aug_batch_size, -1))

        return result_image_batch, result_label_batch
