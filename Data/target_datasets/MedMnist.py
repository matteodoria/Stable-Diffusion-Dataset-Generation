from Data.target_datasets.Dataset import Dataset
import numpy as np
import tensorflow as tf

class PathMNIST(Dataset):
    def __init__(self):
        super().__init__("path_mnist", 9, 28)

    def get_labels(self):
        labels = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
        return labels

    def load_data_real(self, batch_size = None, return_raw = False, categorical = True, shuffle = True, seed = 1234):
        dataset_dict = np.load("Data/MNIST/pathmnist.npz")
        x_train = dataset_dict["train_images"]
        y_train = dataset_dict["train_labels"]
        x_valid = dataset_dict["val_images"]
        y_valid = dataset_dict["val_labels"]
        x_test = dataset_dict["test_images"]
        y_test = dataset_dict["test_labels"]

        if return_raw:
            return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

        if categorical:
            y_train = tf.keras.utils.to_categorical(y_train, 9)
            y_valid = tf.keras.utils.to_categorical(y_valid, 9)
            y_test = tf.keras.utils.to_categorical(y_test, 9)

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        if shuffle:
            print("Shuffling")
            train_ds = train_ds.shuffle(buffer_size=40000)

        train_ds = train_ds.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        valid_ds = valid_ds.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, valid_ds, test_ds

    def load_data_syn(self, exp, batch_size = None, shuffle=True, seed=1234, take=None):
        if exp is None:
            raise ValueError("exp must be provided")

        dataset_dict = np.load("Data/MNIST/pathmnist.npz")
        x_valid = dataset_dict["val_images"]
        y_valid = dataset_dict["val_labels"]
        x_test = dataset_dict["test_images"]
        y_test = dataset_dict["test_labels"]

        y_valid = tf.keras.utils.to_categorical(y_valid, 9)
        y_test = tf.keras.utils.to_categorical(y_test, 9)

        valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        if take == 1:
            path = "Data/Synthetic/Exp_0001/pathmnist/90k"
        else:
            # If we need a synthetic dataset larger than the original
            # we use the 10x larger generated dataset
            # and then take a subset of it
            path = "Data/Synthetic/Exp_0001/pathmnist/900k"

        print("Getting data from: ", path)

        train_ds = tf.keras.utils.image_dataset_from_directory(path, label_mode="categorical",
                                                               batch_size=None, image_size=(28, 28), shuffle=True)

        if take != 1:
            train_ds = train_ds.take(take*90000)

        if shuffle:
            b_size = train_ds.cardinality().numpy()
            print("Shuffling with buffer size=", b_size)
            train_ds = train_ds.shuffle(buffer_size=40000, seed=seed)

        train_ds = train_ds.batch(batch_size)

        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        valid_ds = valid_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, valid_ds, test_ds

class DermaMNIST(Dataset):
    def __init__(self):
        super().__init__("derma_mnist", 7, 28)

    def get_labels(self):
        labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
        return labels

    def load_data_syn(self, exp, batch_size = None, shuffle=True, seed=1234, take=None):
        if exp is None:
            raise ValueError("exp must be provided")

        dataset_dict = np.load("Data/MNIST/dermamnist.npz")
        x_valid = dataset_dict["val_images"]
        y_valid = dataset_dict["val_labels"]
        x_test = dataset_dict["test_images"]
        y_test = dataset_dict["test_labels"]

        y_valid = tf.keras.utils.to_categorical(y_valid, 7)
        y_test = tf.keras.utils.to_categorical(y_test, 7)

        valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        if take == 1:
            path = "Data/Synthetic/Exp_0001/dermamnist/7k"
        else:
            # If we need a synthetic dataset larger than the original
            # we use the 10x larger generated dataset
            # and then take a subset of it
            path = "Data/Synthetic/Exp_0001/dermamnist/70k"

        print("Getting data from: ", path)

        train_ds = tf.keras.utils.image_dataset_from_directory(path, label_mode="categorical",
                                                               batch_size=None, image_size=(28, 28), shuffle=True)

        if take != 1:
            train_ds = train_ds.take(take*7000)

        if shuffle:
            b_size = train_ds.cardinality().numpy()
            print("Shuffling with buffer size=", b_size)
            train_ds = train_ds.shuffle(buffer_size=b_size, seed=seed)

        train_ds = train_ds.batch(batch_size)

        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        valid_ds = valid_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, valid_ds, test_ds

    def load_data_real(self, batch_size = None, return_raw = False, categorical = True, shuffle = True, seed = 1234):
        dataset_dict = np.load("Data/MNIST/dermamnist.npz")
        x_train = dataset_dict["train_images"]
        y_train = dataset_dict["train_labels"]
        x_valid = dataset_dict["val_images"]
        y_valid = dataset_dict["val_labels"]
        x_test = dataset_dict["test_images"]
        y_test = dataset_dict["test_labels"]

        if return_raw:
            return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

        if categorical:
            y_train = tf.keras.utils.to_categorical(y_train, 7)
            y_valid = tf.keras.utils.to_categorical(y_valid, 7)
            y_test = tf.keras.utils.to_categorical(y_test, 7)

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        if shuffle:
            print("Shuffling")
            train_ds = train_ds.shuffle(buffer_size=7007)

        train_ds = train_ds.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        valid_ds = valid_ds.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, valid_ds, test_ds
    
class RetinaMNIST(Dataset):
    def __init__(self):
        super().__init__("retina_mnist", 5, 28)

    def get_labels(self):
        labels = ['Grade0', 'Grade1', 'Grade2', 'Grade3', 'Grade4']
        return labels

    def load_data_syn(self, exp, batch_size = None, shuffle=True, seed=1234, take=None):
        if exp is None:
            raise ValueError("exp must be provided")

        dataset_dict = np.load("Data/MNIST/retinamnist.npz")
        x_valid = dataset_dict["val_images"]
        y_valid = dataset_dict["val_labels"]
        x_test = dataset_dict["test_images"]
        y_test = dataset_dict["test_labels"]

        y_valid = tf.keras.utils.to_categorical(y_valid, 5)
        y_test = tf.keras.utils.to_categorical(y_test, 5)

        valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        if take == 1:
            path = "Data/Synthetic/Exp_0001/retinamnist/1k"
        else:
            # If we need a synthetic dataset larger than the original
            # we use the 10x larger generated dataset
            # and then take a subset of it
            path = "Data/Synthetic/Exp_0001/retinamnist/10k"

        print("Getting data from: ", path)

        train_ds = tf.keras.utils.image_dataset_from_directory(path, label_mode="categorical",
                                                               batch_size=None, image_size=(28, 28), shuffle=True)

        if take != 1:
            train_ds = train_ds.take(take*1080)

        if shuffle:
            b_size = train_ds.cardinality().numpy()
            print("Shuffling with buffer size=", b_size)
            train_ds = train_ds.shuffle(buffer_size=b_size, seed=seed)

        train_ds = train_ds.batch(batch_size)

        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        valid_ds = valid_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, valid_ds, test_ds

    def load_data_real(self, batch_size = None, return_raw = False, categorical = True, shuffle = True, seed = 1234):
        dataset_dict = np.load("Data/MNIST/retinamnist.npz")
        x_train = dataset_dict["train_images"]
        y_train = dataset_dict["train_labels"]
        x_valid = dataset_dict["val_images"]
        y_valid = dataset_dict["val_labels"]
        x_test = dataset_dict["test_images"]
        y_test = dataset_dict["test_labels"]

        if return_raw:
            return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

        if categorical:
            y_train = tf.keras.utils.to_categorical(y_train, 5)
            y_valid = tf.keras.utils.to_categorical(y_valid, 5)
            y_test = tf.keras.utils.to_categorical(y_test, 5)

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        if shuffle:
            print("Shuffling")
            train_ds = train_ds.shuffle(buffer_size=1080)

        train_ds = train_ds.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        valid_ds = valid_ds.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, valid_ds, test_ds


class BloodMNIST(Dataset):
    def __init__(self):
        super().__init__("blood_mnist", 8, 28)

    def get_labels(self):
        labels = ["neutrophils", "eosinophils", "basophils", "lymphocytes", "monocytes",
                  "immature granulocytes", "erythroblasts", "platelets"]
        return labels

    def load_data_syn(self, exp, batch_size = None, shuffle=True, seed=1234, take=None):
        if exp is None:
            raise ValueError("exp must be provided")

        dataset_dict = np.load("Data/MNIST/bloodmnist.npz")
        x_valid = dataset_dict["val_images"]
        y_valid = dataset_dict["val_labels"]
        x_test = dataset_dict["test_images"]
        y_test = dataset_dict["test_labels"]

        y_valid = tf.keras.utils.to_categorical(y_valid, 8)
        y_test = tf.keras.utils.to_categorical(y_test, 8)

        valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        if take == 1:
            path = "Data/Synthetic/Exp_0001/bloodmnist/12k"
        else:
            # If we need a synthetic dataset larger than the original
            # we use the 10x larger generated dataset
            # and then take a subset of it
            path = "Data/Synthetic/Exp_0001/bloodmnist/120k"

        print("Getting data from: ", path)

        train_ds = tf.keras.utils.image_dataset_from_directory(path, label_mode="categorical",
                                                               batch_size=None, image_size=(28, 28), shuffle=True)

        if take != 1:
            train_ds = train_ds.take(take*12000)

        if shuffle:
            b_size = train_ds.cardinality().numpy()
            print("Shuffling with buffer size=", 12000)
            train_ds = train_ds.shuffle(buffer_size=b_size, seed=seed)

        train_ds = train_ds.batch(batch_size)

        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        valid_ds = valid_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, valid_ds, test_ds

    def load_data_real(self, batch_size = None, return_raw = False, categorical = True, shuffle = True, seed = 1234):
        dataset_dict = np.load("Data/MNIST/bloodmnist.npz")
        x_train = dataset_dict["train_images"]
        y_train = dataset_dict["train_labels"]
        x_valid = dataset_dict["val_images"]
        y_valid = dataset_dict["val_labels"]
        x_test = dataset_dict["test_images"]
        y_test = dataset_dict["test_labels"]

        if return_raw:
            return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

        if categorical:
            y_train = tf.keras.utils.to_categorical(y_train, 8)
            y_valid = tf.keras.utils.to_categorical(y_valid, 8)
            y_test = tf.keras.utils.to_categorical(y_test, 8)

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        if shuffle:
            print("Shuffling")
            train_ds = train_ds.shuffle(buffer_size=11959)

        train_ds = train_ds.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        valid_ds = valid_ds.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, valid_ds, test_ds
