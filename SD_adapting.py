import keras.models
import tensorflow as tf
from keras_cv.models.stable_diffusion import NoiseScheduler
import os
import argparse
from Models.stable_diffusion import MyStableDiffusion
from Data.target_datasets import Cifar100, Cifar10, MedMnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEED = 1234

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default='cifar10', type=str, metavar='NAME',
                        choices=["cifar10", "cifar100", "pathmnist", "dermamnist", "bloodmnist", "retinamnist"],
                        help="The target dataset for adapting Stable Diffusion"
                             "Choose from: cifar10, cifar100, pathmnist, dermamnist, bloodmnist, or retinamnist.")
    parser.add_argument("--train_model", default="enc", type=str, choices=["enc", "dif"],
                        help="The model component to train: 'enc' for text encoder, 'dif' for diffusion model.")
    parser.add_argument("--enc_epoch", default=0, type=int,
                        help="Epoch number from which to load pre-trained encoder weights. If 0, start training from scratch.")
    parser.add_argument("--dif_epoch", default=0, type=int,
                        help="Epoch number from which to load pre-trained diffusion model weights. If 0, start training from scratch.")
    parser.add_argument("--exp", type=str, required=True,
                        help="Experiment identifier (e.g., '0001'). This is used to name the directory where weights are saved.")
    parser.add_argument("--freq", default=1, type=int,
                        help="Frequency (in epochs) at which to save model weights during training.")

    return parser.parse_args()

def load_dataset(name: str, batch_size: int = None, return_raw=False):
    """
        Loads a specified dataset.

        Args:
            name (str): The name of the dataset to load. Supported options are:
                - 'cifar10'
                - 'cifar100'
                - 'pathmnist'
                - 'dermamnist'
                - 'bloodmnist'
                - 'retinamnist'
            batch_size (int, optional): If provided, loads the data in batches of this size. If None, loads the entire dataset at once. Defaults to None.
            return_raw (bool, optional): If True, returns the raw dataset as a tuple of (x_train, y_train), resolution, and labels. If False, returns a preprocessed dataset iterator, resolution, and labels. Defaults to False.

        Returns:
            If return_raw is False:
                train_ds: A preprocessed dataset iterator for training.
                resolution: A tuple (height, width) representing the image resolution.
                labels: A list of class labels.
            If return_raw is True:
                (x_train, y_train): A tuple of numpy arrays representing the training data and labels.
                resolution: A tuple (height, width) representing the image resolution.
                labels: A list of class labels.

        Raises:
            ValueError: If an invalid dataset name is provided.
        """
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

    resolution = ds.get_resolution()
    labels = ds.get_labels()
    if not return_raw:
        train_ds, _, _ = ds.load_data(batch_size=batch_size, seed=SEED, return_raw=return_raw, shuffle=True)
        return train_ds, resolution, labels
    else:
        (x_train, y_train), (_, _) = ds.load_data(batch_size=batch_size, seed=SEED, return_raw=return_raw)
        return (x_train, y_train), resolution, labels

class SaveWeights(keras.callbacks.Callback):
    """
        Keras callback for periodically saving model weights during training.

        This callback extends the base `keras.callbacks.Callback` class to provide functionality
        for saving the weights of a specified model at regular intervals during training.

        Attributes:
            ft_model (tf.keras.Model): The Keras model whose weights will be saved.
            freq (int): The frequency (in epochs) at which to save weights.
            path (str): The file path template for saving weights (e.g., "weights_{epoch}.h5").
            last_epoch (int): The last epoch from which to resume saving (useful for restarting training).

        Methods:
            on_epoch_end(epoch, logs=None):
                Callback function executed at the end of each training epoch. Saves the model weights
                if the current epoch matches the specified frequency or is the first epoch.

            set_last_epoch(new_last_epoch):
                Updates the `last_epoch` attribute, allowing you to resume saving from a specific epoch.
    """
    def __init__(self, ft_model, freq, path, last_epoch, **kwargs):
        super().__init__(**kwargs)
        self.ft_model = ft_model
        self.freq = freq
        self.path = path
        self.last_epoch = last_epoch

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        if (self.last_epoch+epoch) % self.freq == 0 or epoch == 1:
            print("... Saving weights ...")
            self.ft_model.save_weights(filepath=self.path.format(epoch=self.last_epoch+epoch))
            print("DONE")

    def set_last_epoch(self, new_last_epoch):
        self.last_epoch = new_last_epoch

class StableDiffusionFineTuner(tf.keras.Model):
    """
        Keras model for fine-tuning Stable Diffusion on custom image-text pairs.

        This class extends the `tf.keras.Model` base class and provides a customized training loop
        for fine-tuning either the text encoder or the diffusion model of a pre-trained Stable
        Diffusion model.

        Attributes:
            stable_diffusion (StableDiffusionModel): The pre-trained Stable Diffusion model to fine-tune.
            ft_model (str): Specifies the model component to fine-tune ("enc" for text encoder, "dif" for diffusion model).
            res (int): The image resolution for training (minimum 128).
            num_classes (int): The number of classes in the dataset.
            training_image_encoder (keras.Model): Submodel extracted from the Stable Diffusion image encoder for training.
            noise_scheduler (NoiseScheduler): Noise scheduler used for adding noise to latents during training.

        Methods:
            sample_from_encoder_outputs(outputs): Samples latents from the output of the image encoder.
            train_step(data):
                Performs one training step:
                    1. Encodes images to latents and adds noise.
                    2. Encodes text labels into conditional vectors.
                    3. Generates timestep embeddings.
                    4. Predicts noise using the diffusion model.
                    5. Calculates the loss.
                    6. Computes gradients and updates the trainable weights of the specified model component.

        Raises:
            ValueError: If an invalid value for `ft_model` is provided.
    """

    def __init__(self, stable_diffusion, ft_model, res, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.stable_diffusion = stable_diffusion
        self.ft_model = ft_model

        self.training_image_encoder = keras.Model(
            self.stable_diffusion.image_encoder.input,
            self.stable_diffusion.image_encoder.layers[-2].output,
        )

        self.noise_scheduler = NoiseScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            train_timesteps=1000,
        )
        if res < 128:
            res = 128
        self.res = res
        self.num_classes = num_classes

    def sample_from_encoder_outputs(self, outputs):
        mean, logvar = tf.split(outputs, 2, axis=-1)
        logvar = tf.clip_by_value(logvar, -30.0, 20.0)
        std = tf.exp(0.5 * logvar)
        sample = tf.random.normal(tf.shape(mean))
        return mean + std * sample

    def train_step(self, data):
        images, labels = data

        with tf.GradientTape() as tape:
            latents = self.sample_from_encoder_outputs(self.training_image_encoder(images))
            latents = latents * 0.18215

            noise = tf.random.normal(tf.shape(latents))
            batch_dim = tf.shape(latents)[0]

            timesteps = tf.random.uniform(
                (batch_dim,),
                minval=0,
                maxval=self.noise_scheduler.train_timesteps,
                dtype=tf.int64,
            )

            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            conditional_vector = self.stable_diffusion.text_encoder(labels)

            timestep_embeddings = tf.map_fn(
                fn=self.stable_diffusion.get_timestep_embedding,
                elems=timesteps,
                fn_output_signature=tf.float32,
            )

            noise_pred = self.stable_diffusion.diffusion_model(
                [noisy_latents, timestep_embeddings, conditional_vector]
            )

            # Compute the loss
            loss = self.compiled_loss(noise_pred, noise)
            loss = tf.reduce_mean(loss, axis=2)
            loss = tf.reduce_mean(loss, axis=1)
            loss = tf.reduce_mean(loss)

        # Load the trainable weights and compute the gradients for them
        if self.ft_model == 'enc':
            trainable_weights = self.stable_diffusion.text_encoder.trainable_weights
        elif self.ft_model == 'dif':
            trainable_weights = self.stable_diffusion.diffusion_model.trainable_weights
        else:
            raise "No Model to fine tune"

        grads = tape.gradient(loss, trainable_weights)

        self.optimizer.apply_gradients(zip(grads, trainable_weights))

        return {"loss": loss}

def main(args):
    # PARAMETERS
    dataset_name = args.dataset
    train_model = args.train_model
    experiment_code = args.exp

    if train_model == 'enc':
        batch_size = 64
        epochs = 50
        lr = 1e-4
        print("\nTransfer Learning of the Class Encoder:\n"
              f"- batch size = {batch_size}\n"
              f"- epochs = {epochs}\n"
              f"- learning rate = {lr}\n")
    elif train_model == 'dif':
        batch_size = 16
        epochs = 10
        lr = 1e-5
        print("\nFine tuning of the Diffusion Model:\n"
              f"- batch size = {batch_size}\n"
              f"- epochs = {epochs}\n"
              f"- learning rate = {lr}\n")
    else:
        print("Argument 'train_model' must be one between 'enc' and 'dif'!")
        return -1

    ## 1. Loading Dataset
    print("... LOADING DATA FROM {} ...".format(dataset_name.upper()))
    train_ds, res, labels = load_dataset(dataset_name, batch_size=batch_size, return_raw=False)

    print("Resolution:", res)
    num_classes = len(labels)
    print("#Classes:", num_classes)
    ds_len = train_ds.cardinality().numpy()
    print("Dataset length:", ds_len)

    if res < 128:   # 128 is the minimum working resolution for Keras Stable Diffusion
        train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, [128, 128]), y))
    train_ds = train_ds.map(lambda x, y: (x / 127.5 - 1, y))

    print("+++ DATASET {} LOADED SUCCESSFULLY +++".format(dataset_name))

    ## 2. Load the New Stable Diffusion Model
    print("\n... LOADING STABLE DIFFUSION MODEL ...")

    enc_folder_path = f"Checkpoints/DDPM/Exp_{experiment_code}/{dataset_name}/MyEmbedding/"
    dif_folder_path = f"Checkpoints/DDPM/Exp_{experiment_code}/{dataset_name}/DiffusionFt/"

    os.makedirs(enc_folder_path, exist_ok=True)
    os.makedirs(dif_folder_path, exist_ok=True)

    enc_epoch = args.enc_epoch
    dif_epoch = args.dif_epoch

    enc_model_path = enc_folder_path + "epoch{epoch}.hdf5"
    diff_model_path = dif_folder_path + "epoch{epoch}.hdf5"

    model = MyStableDiffusion(res=res, num_classes=len(labels),
                              original_diffusion=True, original_text_encoder=False,
                              enc_weight_path=enc_model_path.format(epoch=enc_epoch),
                              diff_weight_path=diff_model_path.format(epoch=dif_epoch))

    print("+++ STABLE DIFFUSION MODEL LOADED SUCCESSFULLY +++")

    print("--------------------------------------------------------------------------------\n\n"
          "TRAINING PHASE")

    ## 3. Preparing the Fine Tuner
    print("... LOADING STABLE DIFFUSION ADAPTER ...")
    trainer = StableDiffusionFineTuner(stable_diffusion=model, ft_model=train_model,
                                       res=res, num_classes=num_classes, name='trainer')

    model.set_train_mode(train_model=train_model)

    learning_rate = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr, decay_steps=ds_len * epochs
    )
    optimizer = tf.keras.optimizers.Adam(
        weight_decay=0.004, learning_rate=learning_rate, epsilon=1e-8, global_clipnorm=10
    )

    trainer.compile(
        optimizer=optimizer,
        # We are performing reduction manually in our train step, so none is required here.
        loss=tf.keras.losses.MeanSquaredError(reduction="none"),
    )

    os.makedirs(dif_folder_path, exist_ok=True)

    callbacks = [
        SaveWeights(ft_model=model.text_encoder if train_model == 'enc' else model.diffusion_model,
                    path=enc_model_path if train_model == 'enc' else diff_model_path,
                    freq=args.freq,
                    last_epoch=enc_epoch if train_model == 'enc' else dif_epoch),
        ]

    print("+++ STABLE DIFFUSION ADAPTER LOADED SUCCESSFULLY +++")

    ## 5. Training
    print("... TRAINING ...")
    trainer.fit(train_ds, epochs=epochs, callbacks=callbacks)

    print("+++ STABLE DIFFUSION ADAPTATION PHASE COMPLETED +++")


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    args = parse_args()
    main(args)
