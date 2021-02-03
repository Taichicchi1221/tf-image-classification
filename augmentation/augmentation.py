from dataclasses import dataclass
import tensorflow as tf

@dataclass
class SingleImageAugmentator:
    seed: int = 42
    RANDOM_FLIP_LEFT_RIGHT: bool = True
    RANDOM_FLIP_UP_DOWN: bool = True
    RANDOM_BRIGHTNESS: bool = True
    RANDOM_BRIGHTNESS_MAX_DELTA: float = 0.2
    RANDOM_CONTRAST: bool = True
    RANDOM_CONTRAST_LOWER: float = 0.6
    RANDOM_CONTRAST_UPPER: float = 1.4
    RANDOM_HUE: bool = True
    RANDOM_HUE_MAX_DELTA: float = 0.1
    RANDOM_SATURATION: bool = True
    RANDOM_SATURATION_LOWER: float = 0.5
    RANDOM_SATURATION_UPPER: float = 1.5
    RANDOM_BLOCKOUT: bool = True
    RANDOM_BLOCKOUT_SL: float = 0.1
    RANDOM_BLOCKOUT_SH: float = 0.2
    RANDOM_BLOCKOUT_RL: float = 0.4
    
    def _random_blockout(
        self,
        img,
        sl=0.1,
        sh=0.2,
        rl=0.4,
    ):
        h, w, c = img.shape
        origin_area = tf.cast(h*w, tf.float32)

        e_size_l = tf.cast(tf.round(tf.sqrt(origin_area * sl * rl)), tf.int32)
        e_size_h = tf.cast(tf.round(tf.sqrt(origin_area * sh / rl)), tf.int32)

        e_height_h = tf.minimum(e_size_h, h)
        e_width_h = tf.minimum(e_size_h, w)

        erase_height = tf.random.uniform(shape=[], minval=e_size_l, maxval=e_height_h, dtype=tf.int32)
        erase_width = tf.random.uniform(shape=[], minval=e_size_l, maxval=e_width_h, dtype=tf.int32)

        erase_area = tf.zeros(shape=[erase_height, erase_width, c])
        erase_area = tf.cast(erase_area, tf.uint8)

        pad_h = h - erase_height
        pad_top = tf.random.uniform(shape=[], minval=0, maxval=pad_h, dtype=tf.int32)
        pad_bottom = pad_h - pad_top

        pad_w = w - erase_width
        pad_left = tf.random.uniform(shape=[], minval=0, maxval=pad_w, dtype=tf.int32)
        pad_right = pad_w - pad_left

        erase_mask = tf.pad([erase_area], [[0,0],[pad_top, pad_bottom], [pad_left, pad_right], [0,0]], constant_values=1)
        erase_mask = tf.squeeze(erase_mask, axis=0)
        erased_img = tf.multiply(tf.cast(img,tf.float32), tf.cast(erase_mask, tf.float32))

        return tf.cast(erased_img, img.dtype)
        
    def __call__(
        self,
        image,
        label = None,
    ):
        if self.RANDOM_FLIP_LEFT_RIGHT: image = tf.image.random_flip_left_right(image, seed = self.seed)
        if self.RANDOM_FLIP_UP_DOWN: image = tf.image.random_flip_up_down(image, seed = self.seed)
        if self.RANDOM_BRIGHTNESS: image = tf.image.random_brightness(image, self.RANDOM_BRIGHTNESS_MAX_DELTA, seed = self.seed)
        if self.RANDOM_CONTRAST: image = tf.image.random_contrast(image, self.RANDOM_CONTRAST_LOWER, self.RANDOM_CONTRAST_UPPER, seed = self.seed)
        if self.RANDOM_HUE: image = tf.image.random_hue(image, self.RANDOM_HUE_MAX_DELTA, seed = self.seed)
        if self.RANDOM_SATURATION: image = tf.image.random_saturation(image, self.RANDOM_SATURATION_LOWER, self.RANDOM_SATURATION_UPPER, seed = self.seed)
        if self.RANDOM_BLOCKOUT: image = self._random_blockout(image, self.RANDOM_BLOCKOUT_SL, self.RANDOM_BLOCKOUT_SH, self.RANDOM_BLOCKOUT_RL)
        
        return image, label

@dataclass
class MixImageAugmentator:
    seed: int = 42
    AUG_BATCH: int = 32
    IMAGE_SIZE_0: int = 256
    IMAGE_SIZE_1: int = 256
    CHANNELS: int = 3
    CLASSES: int = 1
    MIXUP_PROB: float = 0.00
    MIXUP_ALPHA: float = 1.0
    CUTMIX_PROB: float = 0.00
    CUTMIX_ALPHA: float = 1.0
    FMIX_PROB: float = 0.00
    FMIX_ALPHA: float = 1.0
    FMIX_DECAY: float = 3.0
    
    def __post_init__(self):
        self.IMAGE_MAX_W_H = max(self.IMAGE_SIZE_0, self.IMAGE_SIZE_1)
        self.IMAGE_PIXEL_COUNT = self.IMAGE_SIZE_0 * self.IMAGE_SIZE_1
    
    def _beta_sampling(self, shape, alpha = 1.0):
        r1 = tf.random.gamma(shape, alpha, 1, dtype = tf.float32)
        r2 = tf.random.gamma(shape, alpha, 1, dtype = tf.float32)
        return r1 / (r1 + r2)
    
    def _calc_mixup(
        self,
        alpha = 1.0,
        PROBABILITY = 1.0,
    ):
        P = tf.cast(tf.random.uniform([self.AUG_BATCH], 0, 1) <= PROBABILITY, tf.float32)
        a = self._beta_sampling([self.AUG_BATCH], alpha) * P
        return a, a

    def _batch_mixup(
        self,
        image_batch,
        label_batch,
        labeled = True,
        alpha = 1.0,
        PROBABILITY = 1.0,
    ):
        imgs = []; labs = []
        image_mix_ratios, label_mix_ratios = self._calc_mixup(alpha, PROBABILITY)
        
        for j in range(self.AUG_BATCH):
            k = tf.cast( tf.random.uniform([], 0, self.AUG_BATCH),tf.int32)
            img1 = image_batch[j,]
            img2 = image_batch[k,]
            lab1 = label_batch[j,]
            lab2 = label_batch[k,]
        
            result_image = image_mix_ratios[j, ] * img1 + (1 - image_mix_ratios[j, ]) * img2
            imgs.append(result_image)
            if labeled:
                result_label = label_mix_ratios[j, ] * lab1 + (1 - label_mix_ratios[j, ]) * lab2
                labs.append(result_label)

        result_image_batch = tf.reshape(tf.stack(imgs),(self.AUG_BATCH, self.IMAGE_SIZE_0, self.IMAGE_SIZE_1, self.CHANNELS))
        if labeled:
            result_label_batch = tf.reshape(tf.stack(labs),(self.AUG_BATCH, self.CLASSES))
        else:
            result_label_batch = tf.reshape(label_batch, (self.AUG_BATCH, ))

        return result_image_batch, result_label_batch
    
    def _calc_cutmix(
        self,
        alpha = 1.0,
        PROBABILITY = 1.0,
    ):
        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
        P = tf.cast( tf.random.uniform([self.AUG_BATCH],0,1)<=PROBABILITY, tf.int32)
        # CHOOSE RANDOM LOCATION
        x = tf.cast( tf.random.uniform([self.AUG_BATCH],0,self.IMAGE_SIZE_1),tf.int32)
        y = tf.cast( tf.random.uniform([self.AUG_BATCH],0,self.IMAGE_SIZE_0),tf.int32)
        b = self._beta_sampling([self.AUG_BATCH], alpha) # this is beta dist with alpha=1.0
        WIDTH = tf.cast(self.IMAGE_SIZE_1 * tf.math.sqrt(1 - b),tf.int32) * P
        HEIGHT = tf.cast(self.IMAGE_SIZE_0 * tf.math.sqrt(1 - b),tf.int32) * P
        
        ya = tf.math.maximum(0, y - WIDTH//2)
        yb = tf.math.minimum(self.IMAGE_SIZE_1, y + WIDTH // 2)
        xa = tf.math.maximum(0, x - WIDTH//2)
        xb = tf.math.minimum(self.IMAGE_SIZE_0, x + WIDTH // 2)
        # MAKE CUTMIX RATIO
        image_ratios = []
        label_ratios = []
        for j in range(self.AUG_BATCH):
            one = tf.ones([yb[j,] - ya[j,], xa[j,]], dtype = tf.float32)
            two = tf.zeros([yb[j,] - ya[j,], xb[j,] - xa[j,]], dtype = tf.float32)
            three = tf.ones([yb[j,] - ya[j,], self.IMAGE_SIZE_1 - xb[j,]], dtype = tf.float32)
            middle = tf.concat([one, two, three], axis = 1)
            res_image_ratio = tf.concat(
                [
                    tf.ones([ya[j,], self.IMAGE_SIZE_1], dtype = tf.float32),
                    middle,
                    tf.ones([self.IMAGE_SIZE_0 - yb[j,], self.IMAGE_SIZE_1], dtype = tf.float32)
                ],axis=0
            )
            image_ratios.append(tf.expand_dims(res_image_ratio, -1))
            
            # MAKE CUTMIX LABEL
            a = 1 - tf.cast(HEIGHT[j, ] * WIDTH[j, ] / self.IMAGE_SIZE_0 / self.IMAGE_SIZE_1, tf.float32)
            label_ratios.append(a)

        return tf.stack(image_ratios), tf.stack(label_ratios)
    
    def _batch_cutmix(
        self,
        image_batch,
        label_batch,
        labeled = True,
        alpha = 1.0,
        PROBABILITY = 1.0,
    ):
        imgs = []; labs = []
        image_mix_ratios, label_mix_ratios = self._calc_cutmix(alpha, PROBABILITY)
        for j in range(self.AUG_BATCH):
            k = tf.cast( tf.random.uniform([], 0, self.AUG_BATCH),tf.int32)
            img1 = image_batch[j,]
            img2 = image_batch[k,]
            lab1 = label_batch[j,]
            lab2 = label_batch[k,]

            result_image = image_mix_ratios[j, ] * img1 + (1 - image_mix_ratios[j, ]) * img2
            imgs.append(result_image)
            if labeled:
                result_label = label_mix_ratios[j, ] * lab1 + (1 - label_mix_ratios[j, ]) * lab2
                labs.append(result_label)

        result_image_batch = tf.reshape(tf.stack(imgs),(self.AUG_BATCH, self.IMAGE_SIZE_0, self.IMAGE_SIZE_1, self.CHANNELS))
        if labeled:
            result_label_batch = tf.reshape(tf.stack(labs),(self.AUG_BATCH, self.CLASSES))
        else:
            result_label_batch = tf.reshape(label_batch, (self.AUG_BATCH, ))
            
        return result_image_batch, result_label_batch
    
    # https://github.com/ecs-vlc/FMix/blob/master/fmix.py
    def _fftfreq(self, n, d=1.0):
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
        return tf.math.sqrt(fx_square[ tf.newaxis, : ] + fy_square[ : , tf.newaxis ])
    
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
        scale_1hw1 = tf.expand_dims(scale_hw, -1)[ tf.newaxis, : ]
        spectrum_bhw2 = scale_1hw1 * param_bhw2
        return spectrum_bhw2

    def _make_low_freq_images(self, data_count, decay):
        # Make a mask image by inverse Fourier transform of a spectrum,
        # which is generated by self._get_spectrum().
        freqs = self._fftfreqnd(self.IMAGE_SIZE_0, self.IMAGE_SIZE_1)
        spectrum_bhw2 = self._get_spectrum(data_count, freqs, decay)
        spectrum_re_bhw = spectrum_bhw2[:, :, :, 0]
        spectrum_im_bhw = spectrum_bhw2[:, :, :, 1]
        spectrum_comp_bhw = tf.complex(spectrum_re_bhw, spectrum_im_bhw)
        mask_bhw = tf.math.real(tf.signal.ifft2d(spectrum_comp_bhw))

        # Scale the mask values from 0 to 1.
        mask_min_b = tf.reduce_min(mask_bhw, axis=(-2, -1))
        mask_min_b11 = mask_min_b[ :, tf.newaxis, tf.newaxis]
        mask_shift_to_0_bhw = mask_bhw - mask_min_b11
        mask_max_b = tf.reduce_max(mask_shift_to_0_bhw, axis=(-2, -1))
        mask_max_b11 = mask_max_b[ :, tf.newaxis, tf.newaxis]
        mask_scaled_bhw = mask_shift_to_0_bhw / mask_max_b11
        return mask_scaled_bhw

    def _make_binary_masks(self, data_count, low_freq_images_bhw, mix_ratios_b):
        # The goal is "top proportion of the image to have value ‘1’ and the rest to have value ‘0’".
        # To make this I use tf.scatter_nd().  For tf.scatter_nd(), indices and values
        # are necessary.

        # For each image, get indices of an image whose order is sorted from top to bottom.
        # These are used for row indices.  To combine with column indices, they are reshaped to 1D.
        low_freq_images_bp = tf.reshape(low_freq_images_bhw, [data_count, -1])
        row_indices_bp = tf.argsort(low_freq_images_bp, axis=-1, direction='DESCENDING', stable=True)
        row_indices_t = tf.reshape(row_indices_bp, [-1])

        # Make column indices, col_indices_t looks like
        # '[ 0 ... 0 1 ... 1 ..... data_count-1 ... data_count-1]'
        col_indices_b = tf.range(data_count, dtype=tf.int32)
        col_indices_t = tf.repeat(col_indices_b, self.IMAGE_PIXEL_COUNT, axis=-1)

        # Combine column and row indices for scatter_nd.
        scatter_indices_2t = tf.stack([col_indices_t, row_indices_t])
        scatter_indices_t2 = tf.transpose(scatter_indices_2t)

        # Make a tensor which looks like:
        # [[ 0.0 ... 1.0 ]   \  <-- tf.linspace(0.0, 1.0, self.IMAGE_PIXEL_COUNT)
        #   ...               | data_count
        #  [ 0.0 ... 1.0 ]]  /
        linspace_0_1_p = tf.linspace(0.0, 1.0, self.IMAGE_PIXEL_COUNT)
        linspace_0_1_1p = linspace_0_1_p[ tf.newaxis, : ]
        linspace_0_1_bp = tf.repeat(linspace_0_1_1p, data_count, axis=0)

        # Make mix_ratio of the top elements in each data '1' and the rest '0'
        # This looks like:
        # [[ 1.0 1.0 ... 0.0 ]   \    <-- top mix_ratios_b[0] elements are 1.0
        #   ...                   | data_count
        #  [ 1.0 1.0 ... 0.0 ]]  /    <-- top mix_ratios_b[data_count - 1] elements are 1.0
        mix_ratios_b1 = mix_ratios_b[ :, tf.newaxis]
        scatter_updates_bp = tf.where(linspace_0_1_bp <= mix_ratios_b1, 1.0, 0.0)
        scatter_updates_t = tf.reshape(scatter_updates_bp, [-1])

        # Make binary masks by using tf.scatter_nd(), then reshape.
        bin_masks_bp = tf.scatter_nd(scatter_indices_t2, scatter_updates_t, [data_count, self.IMAGE_PIXEL_COUNT])
        bin_masks_bhw1 = tf.reshape(bin_masks_bp, [data_count, self.IMAGE_SIZE_0, self.IMAGE_SIZE_1, 1])
        return bin_masks_bhw1
    
    def _calc_fmix(
        self,
        alpha = 1.0,
        decay = 3.0,
        PROBABILITY = 1.0,
    ):
        # Generate mix ratios by beta distribution.
        mix_ratios = self._beta_sampling([self.AUG_BATCH], alpha = alpha)

        # Generate binary masks, then mix images.
        low_freq_images = self._make_low_freq_images(self.AUG_BATCH, decay)
        bin_masks = self._make_binary_masks(self.AUG_BATCH, low_freq_images, mix_ratios)
        
        return bin_masks, mix_ratios
    
    def _batch_fmix(
        self,
        image_batch,
        label_batch,
        labeled = True,
        alpha = 1.0,
        decay = 3.0,
        PROBABILITY = 1.0,
    ):
        imgs = []; labs = []
        image_mix_ratios, label_mix_ratios = self._calc_fmix(alpha, decay, PROBABILITY)
        for j in range(self.AUG_BATCH):
            k = tf.cast( tf.random.uniform([], 0, self.AUG_BATCH),tf.int32)
            img1 = image_batch[j,]
            img2 = image_batch[k,]
            lab1 = label_batch[j,]
            lab2 = label_batch[k,]

            result_image = image_mix_ratios[j, ] * img1 + (1 - image_mix_ratios[j, ]) * img2
            imgs.append(result_image)
            if labeled:
                result_label = label_mix_ratios[j, ] * lab1 + (1 - label_mix_ratios[j, ]) * lab2
                labs.append(result_label)

        result_image_batch = tf.reshape(tf.stack(imgs),(self.AUG_BATCH, self.IMAGE_SIZE_0, self.IMAGE_SIZE_1, self.CHANNELS))
        if labeled:
            result_label_batch = tf.reshape(tf.stack(labs),(self.AUG_BATCH, self.CLASSES))
        else:
            result_label_batch = tf.reshape(label_batch, (self.AUG_BATCH, ))
        
        return result_image_batch, result_label_batch
        

    def __call__(
        self,
        image,
        label,
        labeled = True,
    ):
        """
        image: batch of images whose size is equal to batch augmentation size
        label: batch of labels whose size is equal to batch augmentation size
        """
        image2, label2 = self._batch_mixup(image, label, labeled, self.MIXUP_ALPHA, self.MIXUP_PROB)
        image3, label3 = self._batch_cutmix(image, label, labeled, self.CUTMIX_ALPHA, self.CUTMIX_PROB)
        image4, label4 = self._batch_fmix(image, label, labeled, self.FMIX_ALPHA, self.FMIX_DECAY, self.FMIX_PROB)
        
        imgs = []; labs = []
        P = tf.cast(tf.random.uniform([self.AUG_BATCH], 0, 3),tf.int32)
        P2 = tf.cast(P == 0, tf.float32)
        P3 = tf.cast(P == 1, tf.float32)
        for j in range(self.AUG_BATCH):
            imgs.append(P2[j] * image2[j,] + P3[j] * image3[j,] + (1 - P2[j] - P3[j]) * image4[j,])
            if labeled:
                labs.append(P2[j] * label2[j,] + P3[j] * label3[j,] + (1 - P2[j] - P3[j]) * label4[j,])

        result_image = tf.reshape(tf.stack(imgs), (self.AUG_BATCH, self.IMAGE_SIZE_0, self.IMAGE_SIZE_1, self.CHANNELS))
        if labeled:
            result_label = tf.reshape(tf.stack(labs),(self.AUG_BATCH, self.CLASSES))
        else:
            result_label = tf.reshape(label, (self.AUG_BATCH, ))

        return result_image, result_label