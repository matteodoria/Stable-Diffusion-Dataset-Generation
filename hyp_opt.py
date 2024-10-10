import tensorflow as tf
from Models.ImageClassifiers import resnet
from Models.stable_diffusion import MyStableDiffusion
from Data.target_datasets import Cifar100, Cifar10, MedMnist
import argparse
import optuna
import matplotlib.pyplot as plt
from functools import partial
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
SEED = 1234


class CustomEarlyStopping(tf.keras.callbacks.EarlyStopping):
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

def load_classifier(name: str, res_shape, n_classes):
    match name:
        case 'resnet20':
            filters = 64
            return resnet.ResNet20(input_shape=(res_shape, res_shape, 3), num_classes=n_classes,
                                   initial_filters=filters, seed=SEED)

def load_dataset(name: str, batch_size: int = None, return_raw=False):
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

    (_, _), (x_valid, y_valid), (x_test, y_test) = ds.load_data(batch_size=batch_size, seed=SEED, return_raw=return_raw)
    return x_valid, y_valid, x_test, y_test, ds.get_resolution(), ds.get_labels()


def generate_n_classifier_training(config, ddpm, exp, dataset, res, num_classes, img_total, bs,
                                   train_model, is_f = 50, ugs_f = 7.5, epoch_f = 50):
    """
        Generates images using a DDPM model, trains a classifier on those images, and evaluates its performance.

        This function performs the following steps:

        1. **Hyperparameter Suggestion:**
           - Suggests values for the number of inference steps (`is`), unconditional guidance scale (`ugs`), and epochs (`epoch`)
             using the provided `config` object.
           - The suggested values are based on the input arguments `is_f`, `ugs_f`, and `epoch_f`.

        2. **Weights Loading:**
           - Loads pre-trained weights for either the text encoder or the diffusion model of the DDPM,
             depending on the `train_model` argument.

        3. **Classifier Preparation:**
           - Loads a ResNet20 classifier model.
           - Compiles the classifier with categorical cross-entropy loss (and label smoothing) and Adam optimizer.

        4. **Image Generation:**
           - Generates `img_total` images using the DDPM model.
           - The images are generated in batches of size `bs`, with optional handling for remaining images if
             `img_total` is not divisible by `bs`.
           - Generated images are resized to 28x28 pixels.

        5. **Dataset Creation:**
           - Creates a TensorFlow dataset from the generated images and their corresponding one-hot encoded labels.
           - Shuffles the dataset, rescales image values to [0, 1], batches the data, and prefetches for efficiency.

        6. **Classifier Training:**
           - Trains the classifier on the generated image dataset.
           - Employs early stopping and learning rate reduction callbacks to optimize training.

        7. **Evaluation:**
           - Evaluates the trained classifier on a test dataset (`test_ds`, not shown in this code snippet).
           - Returns the test accuracy.

        Args:
            config (optuna.trial.Trial): An Optuna trial object for hyperparameter suggestion.
            ddpm (MyStableDiffusion): The trained DDPM model used for image generation.
            exp (str): Experiment name or identifier.
            dataset (str): Dataset name.
            res (int): Original image resolution.
            num_classes (int): Number of classes in the dataset.
            img_total (int): Total number of images to generate.
            bs (int): Batch size for image generation.
            train_model (str): Which model to train, either "enc" (text encoder) or "dif" (diffusion model).
            is_f (int, optional): Upper bound for the number of inference steps suggestion (default: 50).
            ugs_f (float, optional): Upper bound for unconditional guidance scale suggestion (default: 7.5).
            epoch_f (int, optional): Upper bound for epoch suggestion (default: 50).

        Returns:
            float: Test accuracy of the trained classifier.
    """

    # 1. Hyper-parameter Suggestion
    _is = config.suggest_int("is", 5, is_f)

    if train_model == 'enc':
        ugs = config.suggest_float('ugs', 0, 7.5)
    elif train_model == 'dif':
        ugs = config.suggest_float('ugs', 0, 2*ugs_f)
    else:
        raise "Model type not supported: either 'enc' or 'dif'"

    epoch = config.suggest_int("epoch", 1, epoch_f)

    print(f"dif_epoch: {epoch} - Is: {_is} - UGS: {ugs}")

    # 2. Weights Loading
    if train_model == 'enc':
        ddpm.text_encoder.load_weights(f"Checkpoints/DDPM/Exp_{exp}/{dataset}/MyEmbedding/epoch{epoch}.hdf5")
    elif train_model == 'dif':
        ddpm.diffusion_model.load_weights(f"Checkpoints/DDPM/Exp_{exp}/{dataset}/DiffusionFt/epoch{epoch}.hdf5")

    # 3. Classifier Preparation
    classifier = load_classifier('resnet20', res, num_classes)

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    classifier.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

    # 4. Image Generation
    img_class = img_total//num_classes

    batch_size_gen = bs
    num_batches = img_total // batch_size_gen
    rest = img_total % batch_size_gen

    label = tf.eye(num_classes)
    label = tf.repeat(label, img_class, axis=0)

    imgs = []
    for i in range(num_batches):
        print(f"\nBatch {i + 1}/{num_batches + 1 if rest != 0 else num_batches}")
        imgs_gen = ddpm.gen(inputs=label[i*batch_size_gen:i*batch_size_gen+batch_size_gen], num_images=batch_size_gen,
                            num_inference_steps=_is, ugs=ugs)
        imgs_gen = tf.image.resize(imgs_gen, [28, 28])
        imgs.extend(imgs_gen)
    if rest != 0:
        imgs_gen = ddpm.gen(inputs=label[-rest:], num_images=rest, num_inference_steps=_is, ugs=ugs)
        imgs_gen = tf.image.resize(imgs_gen, [28, 28])
        imgs.extend(imgs_gen)

    assert len(imgs) == img_total

    # 5. Dataset Creation

    imgs = tf.convert_to_tensor(imgs)
    train_ds = tf.data.Dataset.from_tensor_slices((imgs, label))
    size = train_ds.cardinality().numpy()
    train_ds = train_ds.shuffle(size).map(lambda x, y: (tf.keras.layers.Rescaling(1. / 255)(x), y))
    train_ds = train_ds.batch(256).prefetch(buffer_size=tf.data.AUTOTUNE)

    # 6. Classifier Training
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.1, patience=25, min_lr=1e-5,
                                             mode='max'),
        CustomEarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True),
    ]

    classifier.fit(train_ds, epochs=100, verbose=1, validation_data=valid_ds, callbacks=callbacks)

    # 7. Evaluation
    test_results = classifier.evaluate(test_ds, return_dict=True)
    test_accuracy = test_results["accuracy"]
    print("Test accuracy:", test_accuracy)

    return test_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and Experiment Setup
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "pathmnist", "dermamnist", "bloodmnist", "retinamnist"],
                        help="The dataset to use for training and evaluation.")
    parser.add_argument("--exp", type=str, required=True,
                        help="The experiment number for tracking results.")

    # Training and Optimization
    parser.add_argument("--optimize", action="store_true", default=True,
                        help="Enable hyperparameter optimization (Optuna).")
    parser.add_argument("--no-optimize", action="store_false", dest="optimize",
                        help="Disable hyperparameter optimization (Optuna).")
    parser.add_argument("--train_model", type=str, default="enc", choices=["enc", "dif"],
                        help="The model component to train: 'enc' for text encoder, 'dif' for diffusion model.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for training the classifier.")
    parser.add_argument("--enc_epoch", type=int, default=0,
                        help="Epoch of the Class Encoder for second optimization step.")
    parser.add_argument("--num_imgs", type=int, default=4000,
                        help="Total number of images to generate for training.")

    # Hyperparameters for dif optimization
    parser.add_argument("--is_1", type=int, default=50,
                        help="Number of inference steps during image generation."
                             "Not necessary when 'train_model' is 'enc'")
    parser.add_argument("--ugs_1", type=float, default=7.5,
                        help="Unconditional Guidance Scale value optimized after first step."
                             "Not necessary when 'train_model' is 'enc'")

    # Visualization
    parser.add_argument("--save_plots", action="store_true", help="Save plots of evaluation results.")

    args = parser.parse_args()
    dataset_name = args.dataset
    model = args.train_model
    bs = args.batch_size

    print("---------- HYPER-PARAMETER OPTIMIZATION ----------\n"
          "- Dataset:", dataset_name,
          "\n- Trained Model:", model,
          "\n- Batch Size:", bs
          )

    storage_path = 'sqlite:///' + os.path.abspath(os.getcwd())+"\database.db"
    print("\n- Storage path", storage_path)

    if args.optimize:

        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.HyperbandPruner(),
            direction='maximize',
            study_name=dataset_name + f"_exp{args.exp}_study_{model}",
            load_if_exists=True,
            storage=storage_path,
        )

        print("... LOADING DATASET ...")
        x_valid, y_valid, x_test, y_test, resolution, labels = load_dataset(dataset_name, batch_size=bs, return_raw=True)
        num_classes = len(labels)
        y_valid = tf.keras.utils.to_categorical(y_valid, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        valid_ds = valid_ds.map(lambda x, y: (tf.keras.layers.Rescaling(1. / 255)(x), y))
        valid_ds = valid_ds.batch(256).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.map(lambda x, y: (tf.keras.layers.Rescaling(1. / 255)(x), y))
        test_ds = test_ds.batch(256).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        print("+++ DATASET LOADED SUCCESSFULLY +++")

        print("... LOADING STABLE DIFFUSION ...")

        if model == 'dif':
            enc_weights_path = f"Checkpoints/DDPM/Exp_{args.exp}/{dataset_name}/MyEmbedding/epoch{args.enc_epoch}.hdf5"
        else:
            enc_weights_path = ""

        ddpm = MyStableDiffusion(res=resolution, num_classes=num_classes,
                                 original_diffusion=True, original_text_encoder=False,
                                 enc_weight_path=enc_weights_path,
                                 diff_weight_path="")

        print("+++ STABLE DIFFUSION LOADED SUCCESSFULLY +++")

        num_iter = 50 if model == 'enc' else 40
        is_f = 50 if model == 'enc' else args.is_1
        ugs_f = 7.5 if model == 'enc' else args.ugs_1
        epoch_f = 50 if model == 'enc' else 10

        generate_n_classifier_training = partial(generate_n_classifier_training,
                                                 ddpm = ddpm, exp = args.exp, dataset = dataset_name,
                                                 res = resolution, num_classes = num_classes,
                                                 img_total = args.num_imgs, bs = bs, train_model = model,
                                                 is_f = is_f, ugs_f = ugs_f, epoch_f = epoch_f)
        study.optimize(generate_n_classifier_training, n_trials=num_iter)

    saved_study = optuna.load_study(study_name=f"{dataset_name}_exp{args.exp}_study_{model}", storage=storage_path)
    print("Best trial:", saved_study.best_trial.number)
    print("Best value:", saved_study.best_trial.value)
    print("Best hyperparameters:", saved_study.best_params)

    df = saved_study.trials_dataframe()
    print(df)

    if args.save_plots:
        saved_study = optuna.load_study(study_name=f"{dataset_name}_exp{args.exp}_study_{model}", storage=storage_path)
        save_path = f"optimization_plots/{args.exp}/{dataset_name}/{model}/"
        os.makedirs(save_path, exist_ok=True)

        optuna.visualization.matplotlib.plot_optimization_history(saved_study)
        plt.savefig(save_path+"history.png", bbox_inches='tight')
        optuna.visualization.matplotlib.plot_contour(saved_study, params=["is", "ugs"])
        plt.savefig(save_path+"contour_is_ugs.png", bbox_inches='tight')
        optuna.visualization.matplotlib.plot_contour(saved_study, params=["is", "epoch"])
        plt.savefig(save_path+"contour_is_epoch.png", bbox_inches='tight')
        optuna.visualization.matplotlib.plot_contour(saved_study, params=["epoch", "ugs"])
        plt.savefig(save_path+"contour_epoch_ugs.png", bbox_inches='tight')
        optuna.visualization.matplotlib.plot_intermediate_values(saved_study)
        plt.savefig(save_path+"intermediate_values.png", bbox_inches='tight')
        optuna.visualization.matplotlib.plot_parallel_coordinate(saved_study)
        plt.savefig(save_path+"parallel_coordinate.png", bbox_inches='tight')
        optuna.visualization.matplotlib.plot_param_importances(saved_study)
        plt.savefig(save_path+"param_importances.png", bbox_inches='tight')
        optuna.visualization.matplotlib.plot_rank(saved_study)
        plt.savefig(save_path+"rank.png", bbox_inches='tight')
        optuna.visualization.matplotlib.plot_slice(saved_study)
        plt.savefig(save_path+"slice.png", bbox_inches='tight')
        optuna.visualization.matplotlib.plot_timeline(saved_study)
        plt.savefig(save_path+"timeline.png", bbox_inches='tight')
