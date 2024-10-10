import tensorflow as tf
import os
import argparse
from Models.stable_diffusion import MyStableDiffusion
from Data.target_datasets import Cifar100, Cifar10, MedMnist
from PIL import Image
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
SEED = 1234

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default='cifar10', type=str, metavar='NAME',
                        choices=["cifar10", "cifar100", "pathmnist", "dermamnist", "bloodmnist", "retinamnist"],
                        help="The dataset to use for training and evaluation. Choose from: cifar10, cifar100, "
                             "pathmnist, dermamnist, bloodmnist, or retinamnist.")
    parser.add_argument("--img_class", default=None, type=int, metavar='N',
                        help="Number of images to generate per class. "
                             "If None, calculated automatically based on --img_total and "
                             "the number of classes in the dataset.")
    parser.add_argument("--img_total", default=None, type=int, metavar='T',
                        help="Total number of images to generate.")
    parser.add_argument("--batch_size", default=250, type=int, metavar='BS',
                        help="Batch size for generating images.")
    parser.add_argument("--enc_epoch", default=50, type=int,
                        help="Epoch from which to load the encoder weights.")
    parser.add_argument("--dif_epoch", default=10, type=int,
                        help="Epoch from which to load the diffusion model weights.")
    parser.add_argument("--inf_steps", default=50, type=int,
                        help="Number of inference steps during image generation.")
    parser.add_argument("--ugs", default=7.5, type=float,
                        help="Unconditional Guidance Scale hyper-parameter for image generation.")
    parser.add_argument("--exp", required=True, type=str,
                        help="Experiment identifier (e.g., '0001').")

    return parser.parse_args()

def load_info(dataset):
    match dataset:
        case 'cifar10': return 32, Cifar10.CIFAR10().get_labels()
        case 'cifar100': return 32, Cifar100.CIFAR100().get_labels()
        case 'pathmnist': return 28, MedMnist.PathMNIST().get_labels()
        case 'dermamnist': return 28, MedMnist.DermaMNIST().get_labels()
        case 'bloodmnist': return 28, MedMnist.BloodMNIST().get_labels()
        case 'retinamnist': return 28, MedMnist.RetinaMNIST().get_labels()

def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    args = parse_args()

    # PARAMETERS
    dataset_name = args.dataset
    batch_size = args.batch_size
    img_total = args.img_total
    img_class = args.img_class
    enc_epoch = args.enc_epoch
    dif_epoch = args.dif_epoch
    _is = args.inf_steps
    ugs = args.ugs

    ## 1. Get info of the dataset
    res, labels = load_info(dataset_name)
    num_classes = len(labels)

    if img_total is not None and img_class is not None:
        raise ValueError("Set either num_total or num_per_class. Not both.")
    elif img_total is None:
        img_total = img_class * num_classes
    elif img_class is None:
        img_class = img_total // num_classes
    path_img_total = f"{img_total/1000}k" if img_total > 1000 else img_total

    num_batches = img_total // batch_size
    rest = img_total % batch_size

    ## 2. Load the new Stable Diffusion Model (with Custom Embedding_model)
    print("... LOADING STABLE DIFFUSION MODEL ...")
    emb_weights_path = f"Checkpoints/DDPM/Exp_{args.exp}/{dataset_name}/MyEmbedding/epoch{enc_epoch}.hdf5"
    dif_weights_path = f"Checkpoints/DDPM/Exp_{args.exp}/{dataset_name}/DiffusionFt/epoch{dif_epoch}.hdf5"

    ddpm = MyStableDiffusion(res=res, num_classes=len(labels),
                             original_diffusion=True, original_text_encoder=False,
                             enc_weight_path=emb_weights_path,
                             diff_weight_path=dif_weights_path)

    print("+++ STABLE DIFFUSION MODEL LOADED SUCCESSFULLY +++")

    print("\n\n------------ IMAGE GENERATION ------------"
          "\n- Num classes:", num_classes,
          "\n- Num images per class:", img_class,
          "\n- Inference steps:", _is,
          "\n- Unconditional Guidance Scale:", ugs,
          )

    label = tf.eye(num_classes)
    label = tf.repeat(label, img_class, axis=0)

    save_path = f"Data/Synthetic/Exp_{args.exp}/{dataset_name}/{path_img_total}Enc{enc_epoch}Dif{dif_epoch}Is{_is}Ugs{ugs}/"
    last = {i: 0 for i in range(num_classes)}

    def save_images(images, _labels):
        print(f"Saving {len(images)} images in {save_path} ...")

        for i in range(images.shape[0]):
            img = images[i]
            cls = tf.math.argmax(_labels[i], axis=0)
            image_path = save_path + f"class_{cls:02d}"
            PIL_image = Image.fromarray(img.astype('uint8'), 'RGB')
            os.makedirs(image_path, exist_ok=True)
            n = last[int(cls)]
            PIL_image.save(f"{image_path}/img_{n}.png")
            last[int(cls)] += 1

    ## 3. Generate images
    if num_batches > 0:
        for i in tqdm(range(num_batches)):
            print(f"\nBatch {i + 1}/{num_batches + 1 if rest != 0 else num_batches}")
            label_sel = label[i*batch_size:i*batch_size+batch_size]
            imgs = ddpm.gen(label_sel, num_images=batch_size, num_inference_steps=_is, ugs=ugs)
            save_images(imgs, label_sel)
    if rest != 0:
        imgs = ddpm.gen(label[-rest:], num_images=rest, num_inference_steps=_is, ugs=ugs)
        save_images(imgs, label[-rest:])


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    main()
