## IMPORT
import os
SEED = 1234
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['MPLCONFIGDIR'] = os.getcwd()+'/configs/'
import argparse
import tensorflow as tf
import tensorflow.keras as tfk

tf.autograph.set_verbosity(0)

import warnings
warnings.simplefilter(action='ignore', category=Warning)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import numpy as np
np.random.seed(SEED)

import random
random.seed(SEED)

import logging
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(SEED)
import json
import time
from Data.target_datasets import Cifar100, Cifar10, MedMnist
from Models.ImageClassifiers import resnet
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default='cifar10', type=str, metavar='NAME',
                        choices=["cifar10", "cifar100", "pathmnist", "dermamnist", "bloodmnist", "retinamnist"],
                        help="The dataset to use for training and evaluation.")
    parser.add_argument("--batch_size", default=256, type=int, metavar='BS',
                        help="Batch size for training.")
    parser.add_argument("--epochs", default=100, type=int, metavar='N',
                        help="Number of epochs for training.")
    parser.add_argument("--synthetic", default=True, type=str2bool, metavar='T/F', const=True, nargs='?',
                        help="Use synthetic images for training (T/F).")
    parser.add_argument("--take", default=1, type=int,
                        help="Size of the synthetic dataset with respect to the real one")
    parser.add_argument("--exp", required=True, type=str,
                        help="Experiment identifier (e.g., '0001').")
    parser.add_argument("--classifier", default="resnet20", type=str, metavar='C',
                        help="Classifier architecture to use.")

    parser.add_argument("--train", default=True, type=str2bool, metavar='T/F', const=True, nargs='?',
                        help="Perform training (T/F). If True, the classifier will be trained on the dataset.")
    parser.add_argument("--test", default=True, type=str2bool, metavar='T/F', const=True, nargs='?',
                        help="Perform testing (T/F). If True, the trained classifier will be evaluated on the test set.")

    return parser.parse_args()


class CNNInfoLogger(tf.keras.callbacks.Callback):
    """
    A callback to log training information such as training time, epoch times,
    best performance metrics, model inference time, number of parameters,
    and FLOPs or MACs. It also saves model history and best model, and creates
    a new directory for saving if it doesn't exist.
    """

    def __init__(self, save_path='.', monitor='val_accuracy', save_model=True):
        super(CNNInfoLogger, self).__init__()
        self.save_path = save_path
        self.monitor = monitor
        self.save_model = save_model
        self.start_time = None
        self.end_time = None
        self.epoch_times = []
        self.is_monitoring_accuracy = 'acc' in monitor
        self.best_value = float('-inf') if self.is_monitoring_accuracy else float('inf')
        self.best_epoch = 0
        self.best_model = None
        self.inference_time = None
        self.macs = None

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)

        current_value = logs.get(self.monitor)
        if current_value is not None:
            is_best = self.is_monitoring_accuracy and current_value > self.best_value
            is_best |= not self.is_monitoring_accuracy and current_value < self.best_value

            if is_best:
                self.best_value = current_value
                self.best_epoch = epoch
                self.best_model = self.model

    @staticmethod
    def calculate_MACs(model):
        """
        Estimate the number of Multiply-Accumulate Operations (MACs) for the model.
        This is an approximation and may not be exact for all layer types.
        """
        total_MACs = 0
        for layer in model.layers:
            output_shape = layer.output_shape
            if isinstance(layer, tf.keras.layers.Conv2D):
                # Conv2D layer MACs calculation
                kernel_size = layer.kernel_size
                input_channels = layer.input_shape[-1]
                kernels = layer.filters
                MACs = np.prod(output_shape[1:3]) * input_channels * np.prod(kernel_size) * kernels
                total_MACs += MACs
            elif isinstance(layer, tf.keras.layers.Dense):
                # Dense layer MACs calculation
                input_features = layer.input_shape[-1]
                output_features = layer.units
                MACs = input_features * output_features
                total_MACs += MACs
            elif isinstance(layer, (tf.keras.layers.Activation, tf.keras.layers.MaxPooling2D,
                                    tf.keras.layers.GlobalAveragePooling2D)):
                # These layers do not typically involve MACs, but we can count the output size for simplicity
                MACs = np.prod(output_shape[1:])
                total_MACs += MACs
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                # BatchNormalization involves operations over its parameters, but not typically MACs
                # For simplicity, we can count twice the number of features (for mean and variance)
                MACs = 2 * output_shape[-1]
                total_MACs += MACs
            elif isinstance(layer, tf.keras.layers.Add):
                # Add layer simply adds two tensors, so we can count the output size
                MACs = np.prod(output_shape[1:])
                total_MACs += MACs
            # Add more conditions here for different layer types if necessary

        return total_MACs

    def on_train_end(self, logs=None):
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        average_epoch_time = sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0

        history_path = os.path.join(self.save_path, 'model_history.csv')
        best_model_path = os.path.join(self.save_path, 'best_model.h5')
        training_info_path = os.path.join(self.save_path, 'training_info.csv')

        # Saving model history
        try:
            if hasattr(self.model, 'history') and self.model.history:
                history_df = pd.DataFrame(self.model.history.history)
                history_df.to_csv(history_path, index=False)
                print(f"Model history saved at: {history_path}")
            else:
                print("Model history is empty or not available.")
        except Exception as e:
            print(f"Error saving model history: {e}")

        # Saving best model
        if self.save_model and self.best_model is not None:
            try:
                self.best_model.save(best_model_path)
                print(f"Best model saved at: {best_model_path}")
            except Exception as e:
                print(f"Error saving the best model: {e}")

        try:
            start_inference = time.time()
            _ = self.best_model.predict(np.ones(((1,) + self.model.input_shape[1:])), verbose=0)
            self.inference_time = time.time() - start_inference
        except Exception as e:
            print(f"Error during inference time calculation: {e}")
            self.inference_time = None

        num_parameters = self.model.count_params()
        self.macs = self.calculate_MACs(self.model)

        # Compile and save training information
        info = {
            'Total Training Time (s)': [total_time],
            'Average Epoch Time (s)': [average_epoch_time],
            'Best Epoch': [self.best_epoch],
            'Best Score': [self.best_value],
            'Inference Time (s)': [self.inference_time],
            'Number of Parameters': [num_parameters],
            'Number of MACs': [self.macs]
        }

        try:
            info_df = pd.DataFrame(info)
            info_df.to_csv(training_info_path, index=False)
            print(f"Training information saved at: {training_info_path}")
        except Exception as e:
            print(f"Error saving training information: {e}")

class CustomEarlyStopping(tfk.callbacks.EarlyStopping):
    def __init__(
        self,
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0
    ):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            baseline=baseline,
            restore_best_weights=restore_best_weights
        )
        self.start_from_epoch = start_from_epoch
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        # Only update best weights if the start_from_epoch has been reached
        if epoch >= self.start_from_epoch:
            super().on_epoch_end(epoch, logs)
            # Update best weights manually if they are not set yet
            if self.restore_best_weights and self.best_weights is None:
                self.best_weights = self.model.get_weights()
        else:
            # Keep track of the best weights before start_from_epoch
            current = self.get_monitor_value(logs)
            if self.best_weights is None or self.monitor_op(current - self.min_delta, self.best):
                self.best_weights = self.model.get_weights()
                self.best = current

    def on_train_end(self, logs=None):
        # Restore the best weights if training is finished and restore_best_weights is True
        if self.restore_best_weights and self.best_weights is not None:
            if self.verbose > 0:
                print(f"Restoring model weights from the end of the best epoch.")
            self.model.set_weights(self.best_weights)
        super().on_train_end(logs)

def load_dataset(name: str, exp: str, synthetic: bool = False, batch_size: int = 256, take = None):
    """
        Loads and preprocesses a specified dataset.

        This function loads a dataset, either real or synthetic,
        from a variety of sources, including CIFAR10, CIFAR100 and MedMNIST datasets.

        Args:
            name (str): Name of the dataset to load. Supported options are:
                - "cifar10"
                - "cifar100"
                - "pathmnist"
                - "dermamnist"
                - "bloodmnist"
                - "retinamnist"
            exp (str): Identifier of the experiment
            synthetic (bool, optional): If True, load synthetic images for the dataset. Defaults to False.
            batch_size (int, optional): Batch size for loading the data. Defaults to 256.
            take (int, optional): Number of samples to take per class (if provided). Defaults to None (use the whole dataset).

        Returns:
            tuple: A tuple containing:
                - (train_ds, valid_ds, test_ds): TensorFlow Dataset objects for training, validation, and testing.
                - res: A tuple representing the image resolution (height, width).
                - num_classes: The number of classes in the dataset.

        Raises:
            ValueError: If an unsupported dataset name is provided.
    """
    def preprocess_data(images, labels):
        """Applies preprocessing to the data."""
        inputs = {"images": images, "labels": labels}
        inputs["images"] = tfk.layers.Rescaling(1. / 255)(inputs["images"])

        return inputs["images"], inputs["labels"]

    match name:
        case 'cifar10':
            ds = Cifar10.CIFAR10()
        case 'cifar100':
            ds = Cifar100.CIFAR100()
        case 'pathmnist':
            ds = MedMnist.PathMNIST()
        case 'dermamnist':
            ds = MedMnist.DermaMNIST()
        case 'bloodmnist':
            ds = MedMnist.BloodMNIST()
        case 'retinamnist':
            ds = MedMnist.RetinaMNIST()
        case _:
            raise "Data not found"

    res = ds.get_resolution()
    num_classes = len(ds.get_labels())
    train_ds, valid_ds, test_ds = ds.load_data(batch_size=batch_size, exp=exp, synthetic=synthetic, return_raw=False, categorical=True,
                                               shuffle=True, seed=SEED, take=take)

    train_ds = train_ds.map(lambda x, y: preprocess_data(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds = valid_ds.map(lambda x, y: preprocess_data(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: preprocess_data(x, y), num_parallel_calls=tf.data.AUTOTUNE)

    return (train_ds, valid_ds, test_ds), res, num_classes

def load_classifier(name: str, res_shape, num_classes):
    match name:
        case 'resnet20':
            filters = 64
            return resnet.ResNet20(input_shape=(res_shape, res_shape, 3), num_classes=num_classes,
                                   initial_filters=filters, seed=SEED)

def main(args):
    dataset_name = args.dataset.lower()
    classifier_name = args.classifier
    experiment = args.exp
    batch_size = args.batch_size
    epochs = args.epochs
    if args.synthetic:
        take = args.take
    else:
        take = 1

    classifier_checkpoint_path = f'Checkpoints/Classifiers/Exp_{experiment}/{classifier_name}/'
    os.makedirs(classifier_checkpoint_path, exist_ok=True)

    ## 1. Load results dictionary and specific classifier-dataset-experiment key
    print(f"\n------------ CLASSIFICATION ------------"
          f"\n- classifier: {classifier_name}"
          f"\n- dataset: {dataset_name}"
          f"\n- real/synthetic: {'synthetic' if args.synthetic else 'real'}"
          f"\n- cardinality: x{take}"
          )

    try:
        with open(f"CAS Results/test_accuracies_exp{experiment}.json", 'r') as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        results = {}
        with open(f"CAS Results/test_accuracies_exp{experiment}.json", 'w') as f:
            json.dump(results, f)

    print("\n... LOADING DATASET ...")

    (train_ds, valid_ds, test_ds), res_shape, num_classes = load_dataset(dataset_name,
                                                                         exp = experiment,
                                                                         synthetic = args.synthetic,
                                                                         batch_size = batch_size,
                                                                         take=take)
    try:
        res_class_ds = results[classifier_name]
    except json.JSONDecodeError:
        results[classifier_name] = {}
        res_class_ds = results[classifier_name]

    try:
        res_class_ds = res_class_ds[dataset_name]
    except json.JSONDecodeError:
        res_class_ds[dataset_name] = {}
        res_class_ds = res_class_ds[dataset_name]

    try:
        res_class_ds = res_class_ds[f"x{take}" if args.synthetic else "real"]
    except json.JSONDecodeError:
        res_class_ds[f"x{take}" if args.synthetic else "real"] = {}
        res_class_ds = res_class_ds[f"x{take}" if args.synthetic else "real"]

    print("+++ DATASET LOADED SUCCESSFULLY +++")

    ## 2. Load Classifier
    print(f"\n... LOADING CLASSIFIER: {classifier_name} ...")
    classifier = load_classifier(classifier_name, res_shape, num_classes)

    loss = tfk.losses.CategoricalCrossentropy(label_smoothing=0.1)
    classifier.compile(loss=loss, optimizer='adam',
                       metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5)])

    classifier_weight_path = f"Checkpoints/Classifiers/{classifier_name}/{dataset_name}/"
    classifier_weight_path += f"x{take}/" if args.synthetic else "real/"

    os.makedirs(classifier_weight_path, exist_ok=True)
    classifier_weight_path += 'best_model.h5'

    print(f"+++ CLASSIFIER LOADED SUCCESSFULLY +++")

    print("Train:", args.train)
    if args.train:
        ## TRAIN THE CLASSIFIER
        callbacks = [
            tfk.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.1, patience=10,
                                            min_lr=1e-5, mode='max'),
            CustomEarlyStopping(monitor='val_accuracy', patience=25,
                                restore_best_weights=True),
            CNNInfoLogger(save_path=classifier_weight_path)
        ]

        _ = classifier.fit(train_ds, epochs=epochs, verbose=1, validation_data=valid_ds, callbacks=callbacks)

    print("Test:", args.test)
    if args.test:
        try:
            print("Loading weights from", classifier_weight_path)
            classifier.load_weights(classifier_weight_path)
        except (FileNotFoundError, OSError) as e:
            print(f"No checkpoint found: {e}")
        test_accuracy = classifier.evaluate(test_ds, return_dict=True)
        print("Test results:", test_accuracy)

        print("Saving test result...")

        try:
            res_class_ds["top_1_accuracy"] = test_accuracy['accuracy']
            res_class_ds["top_5_accuracy"] = test_accuracy['top_k_categorical_accuracy']

            with open(f"CAS Results/test_accuracies_exp{experiment}.json", "w") as jsonFile:
                json.dump(results, jsonFile, indent=4)

            print("DONE")
        except (KeyError, IOError, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Something went wrong: {e}")


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    args = parse_args()
    main(args)
