import tensorflow.keras.datasets as datasets
import tensorflow.keras as tfk
import tensorflow as tf
from Data.target_datasets.Dataset import Dataset

class CIFAR10(Dataset):
    def __init__(self):
        super().__init__("cifar10", 10, 32)

    def get_labels(self):
        labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        return labels

    def load_data_real(self, batch_size = None, return_raw = False, categorical = True, shuffle = True, seed = 1234):
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

        if return_raw:
            return (x_train[:40000], y_train[:40000]), (x_train[10000:], y_train[10000:]), (x_test, y_test)

        if categorical:
            y_train = tfk.utils.to_categorical(y_train, 10)
            y_test = tfk.utils.to_categorical(y_test, 10)

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))

        valid_ds = train_ds.skip(40000)
        train_ds = train_ds.take(40000)

        if shuffle:
            print("Shuffling")
            train_ds = train_ds.shuffle(buffer_size=40000, seed=seed)

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

        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        y_train = tfk.utils.to_categorical(y_train, 10)

        if take == 1:
            path = f"Data/Synthetic/Exp_{exp}/cifar10/40k"
        else:
            # If we need a synthetic dataset larger than the original
            # we use the 10x larger generated dataset
            # and then take a subset of it
            path = f"Data/Synthetic/Exp_{exp}/cifar10/400k"

        print("Getting data from: ", path)

        train_ds = tfk.utils.image_dataset_from_directory(path, label_mode="categorical",
                                                          batch_size=None, image_size=(32, 32), shuffle=True)
        if take != 1:
            train_ds = train_ds.take(take*40000)

        if shuffle:
            b_size = train_ds.cardinality().numpy()
            print("Shuffling with buffer size=", b_size)
            train_ds = train_ds.shuffle(buffer_size=40000, seed=seed)

        train_ds = train_ds.batch(batch_size)

        valid_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        valid_ds = valid_ds.skip(40000)

        y_test = tfk.utils.to_categorical(y_test, 10)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        valid_ds = valid_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, valid_ds, test_ds
