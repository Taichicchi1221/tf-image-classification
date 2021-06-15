import tensorflow as tf


class OneOf:
    def __init__(self, object_ls, aug_batch_size) -> None:
        assert len(object_ls) > 1
        self.object_ls = object_ls
        self.l = len(self.object_ls)

    def transform(self, images, labels):
        p = tf.one_hot(
            tf.random.uniform(
                [self.l], 0, self.l, dtype=tf.int32, seed=0
            ), self.l
        )
        imgs = []
        labs = []
        for idx in range(self.l):
            tr_image, tr_label = obj.transform(
                images, labels
            )
            imgs.append(p[idx] * tr_image)
            labs.append(p[idx] * tr_label)

        result_images = tf.stack(imgs)
        result_labels = tf.stack(labs)
        return result_images, result_labels


class Compose:
    def __init__(self, object_ls) -> None:
        self.object_ls = object_ls
        self.l = len(self.object_ls)

    def transform(self, images, labels):
        for obj in self.object_ls:
            images, labels = obj.transform(images, labels)

        return images, labels
