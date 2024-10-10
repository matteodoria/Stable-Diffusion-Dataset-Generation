from Data.target_datasets.Dataset import Dataset
from tensorflow.keras import datasets
import tensorflow.keras as tfk
import tensorflow as tf

class CIFAR100(Dataset):
    def __init__(self):
        super().__init__("cifar10", 100, 32)

    def get_labels(self):
        labels = ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
                  "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
                  "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
                  "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo",
                  "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree",
                  "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
                  "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit",
                  "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper",
                  "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
                  "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe",
                  "whale", "willow_tree", "wolf", "woman", "worm"]

        return labels

    def load_data_real(self, batch_size = None, return_raw = False, categorical = True, shuffle = True, seed = 1234):
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

        if return_raw:
            return (x_train[:40000], y_train[:40000]), (x_train[10000:], y_train[10000:]), (x_test, y_test)

        if categorical:
            y_train = tfk.utils.to_categorical(y_train, 100)
            y_test = tfk.utils.to_categorical(y_test, 100)

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))

        valid_ds = train_ds.skip(40000)
        train_ds = train_ds.take(40000)

        if shuffle:
            print("Shuffling")
            train_ds = train_ds.shuffle(buffer_size=40000)

        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        if batch_size is None:
            return train_ds, valid_ds, test_ds
        else:
            train_ds = train_ds.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            valid_ds = valid_ds.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            test_ds = test_ds.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, valid_ds, test_ds

    def load_data_syn(self, exp, batch_size = None, shuffle=True, seed=1234, take=None):
        if exp is None:
            raise ValueError("exp must be provided")

        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
        x_train = x_train[40000:]
        y_train = y_train[40000:]
        y_train = tfk.utils.to_categorical(y_train, 100)

        if take == 1:
            path = "Data/Synthetic/Exp_0001/cifar100/40k"
        else:
            # If we need a synthetic dataset larger than the original
            # we use the 10x larger generated dataset
            # and then take a subset of it
            path = "Data/Synthetic/Exp_0001/cifar100/400k"

        print("Getting data from: ", path)

        train_ds = tfk.utils.image_dataset_from_directory(path, label_mode="categorical",
                                                          batch_size=None, image_size=(32, 32), shuffle=True)
        if take != 1:
            train_ds = train_ds.take(take * 40000)

        if shuffle:
            b_size = train_ds.cardinality().numpy()
            print("Shuffling with buffer size=", b_size)
            train_ds = train_ds.shuffle(buffer_size=40000, seed=seed)

        train_ds = train_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        valid_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        valid_ds = valid_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        y_test = tfk.utils.to_categorical(y_test, 100)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_ds = test_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, valid_ds, test_ds
