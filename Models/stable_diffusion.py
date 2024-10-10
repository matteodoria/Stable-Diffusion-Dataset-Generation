import tensorflow as tf
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.stable_diffusion import StableDiffusionBase
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder
from keras_cv.backend import ops
from Models.class_encoder import ClassEncoder
import math

## Stable Diffusion (VAE + UNET + Text Encoder + Scheduler)
class MyStableDiffusion(StableDiffusionBase):
    """
        Custom Stable Diffusion model with extended capabilities.

        This class inherits from `StableDiffusionBase` and provides additional features and customization
        options for working with Stable Diffusion models.

        Attributes:
            num_classes (int): The number of classes used for conditional text embeddings.
            img_height, img_width (int): The desired image resolution (minimum 128 for the diffusion model).
            original_diffusion (bool): Whether to use the original diffusion model weights (default: True).
            original_text_encoder (bool): Whether to use the original text encoder weights (default: True).
            enc_weight_path (str): Path to custom text encoder weights (if not using the original).
            dif_weight_path (str): Path to custom diffusion model weights (if not using the original).
            img_height (int): Actual image height used in the model (set to `res` or 128 if `res` < 128).
            img_width (int): Actual image width used in the model (set to `res` or 128 if `res` < 128).

        Methods:
            set_train_mode(train_model):
                Configures the model for training by setting the trainable status of specific components
                (text encoder or diffusion model) based on the `train_model` parameter.
                Prints a summary of the trainable weights for each model component.

            encode_text(prompt):
                Encodes the input prompt(s) using the class encoder and returns the context embeddings.

            _get_unconditional_context():
                Returns unconditional context embeddings, used for generating images without class prompts.

            gen(inputs, neg_input=None, num_images=2, num_inference_steps=50, ugs=7.5, seed=None):
                Generate a 'num_images' number of images given the 'inputs' and the values of
                generation hyperparameters 'num_inference_steps' and 'ugs'.

            text_encoder():
                Overridden method from Stable Diffusion Base.
                If no Encoder is set, it uses either the Original Text Encoder or
                a custom Class Encoder created for our purpose, loading wights if a path is provided.

            diffusion_model():
                Overridden method from Stable Diffusion Base.
                It sets the Original Diffusion Model, setting the correct input context length
                and loading weights, if a path is provided.

            get_pos_ids():
                Generates positional IDs for class embeddings.
                It returns a tensor of shape (1, num_classes) containing positional IDs for class embeddings.
                The values range from 0 to (num_classes - 1).

            def get_timestep_embedding(timestep, dim=320, max_period=10000):
                Generates a sinusoidal positional embedding for a given timestep.
                This method creates a vector representation for the timestep, which is used as additional input
                to the diffusion model. The embedding is constructed using sinusoidal functions of varying
                frequencies, allowing the model to understand the temporal position of the timestep within
                the diffusion process.


        Additional Notes:

            - The diffusion model requires a minimum image resolution of 128x128. If `res` is set lower,
              the model automatically adjusts it to 128.
            - You can customize the text encoder and diffusion model weights by providing paths to
              pre-trained models using the `enc_weight_path` and `diff_weight_path` attributes.

    """
    def __init__(self, num_classes, res,
                 original_diffusion: bool = True, original_text_encoder: bool = True,
                 enc_weight_path: str = "", diff_weight_path: str = ""):
        self.input_res = res
        if res < 128:   # diffusion model doesn't work with resolution < 128
            res = 128
        super().__init__(img_height=res, img_width=res)
        self.img_height = res
        self.img_width = res
        self.num_classes = num_classes
        self.original_diffusion = original_diffusion
        self.original_text_encoder = original_text_encoder
        self.enc_weight_path = enc_weight_path
        self.dif_weight_path = diff_weight_path

        self.diffusion_model.compile()
        self.text_encoder.compile()

    def set_train_mode(self, train_model):
        # Set what is trainable
        self.diffusion_model.trainable = True if train_model == 'dif' else False
        self.decoder.trainable = False
        self.text_encoder.trainable = True if train_model == 'enc' else False

        all_models = [
            self.text_encoder,
            self.diffusion_model,
            self.decoder,
        ]

        print("- Trainable Weights:")
        print([[w.shape for w in model.trainable_weights] for model in all_models])

    def encode_text(self, prompt):
        if not isinstance(prompt, list):
            if len(prompt.shape) == 1:
                prompt = tf.expand_dims(prompt, axis=0)
        context = self.text_encoder(prompt)

        return context

    def _get_unconditional_context(self):
        label = tf.convert_to_tensor([[0]*self.num_classes], dtype=tf.int32)
        unconditional_context = self.text_encoder(label)
        return unconditional_context

    def gen(self, inputs, neg_input=None, num_images=2, num_inference_steps=50, ugs=7.5, seed=None):
        print(f"\n... Generating {num_images} image{'s' if num_images > 1 else ''} ...")
        if len(inputs.shape) == 1:
            inputs = tf.expand_dims(inputs, 0)

        context = self.text_encoder(inputs)

        imgs = self.generate_image(context, negative_prompt=neg_input, batch_size=num_images,
                                   num_steps=num_inference_steps, seed=seed, unconditional_guidance_scale=ugs)

        return imgs

    @property
    def text_encoder(self):
        if self._text_encoder is None:
            if self.original_text_encoder:
                self._text_encoder = TextEncoder(self.num_classes)
            else:
                self._text_encoder = ClassEncoder(max_length=self.num_classes, weight_path=self.enc_weight_path)
            if self.jit_compile:
                self._text_encoder.compile(jit_compile=True)
        return self._text_encoder

    @property
    def diffusion_model(self):
        if self._diffusion_model is None:
            if self.original_diffusion:
                self._diffusion_model = DiffusionModel(
                    self.img_height, self.img_width, max_text_length=100
                )
                try:
                    self._diffusion_model.load_weights(self.dif_weight_path)
                    print("Weights loaded from ", self.dif_weight_path)
                except FileNotFoundError:
                    print("No weights found at specified path. Using HuggingFace weights for Diffusion Model.")
                except Exception as e:
                    print(f"An error occurred while loading weights for Diffusion Model: {e}")
            if self.jit_compile:
                self._diffusion_model.compile(jit_compile=True)
        return self._diffusion_model

    def get_pos_ids(self):
        return ops.expand_dims(ops.arange(self.num_classes, dtype="int32"), 0)

    @staticmethod
    def get_timestep_embedding(timestep, dim=320, max_period=10000):
        half = dim // 2
        freqs = tf.math.exp(
            -math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
        )
        args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
        return embedding
