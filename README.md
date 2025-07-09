# Unrolled networks using PET listmode data

## Environment Setup

To set up the project environment, make sure you have conda installed. 
We recommend to install it from [miniforge](https://github.com/conda-forge/miniforge)

Then run:

```bash
conda env create -f environment.yaml
conda activate pet-lm-dl
```

## Data Preparation Pipeline

1. **Download the BrainWeb PET/MR Dataset**

   Run the following script once to download and extract the dataset into the
   `data/brainweb_petmr_v2`
   ```bash
   python 00_download_brainweb_image_data.py
   ```

2. **Simulate PET Data**

   Use the provided run script to simulate PET listmode data based on the
   downloaded images
   ```bash
   python 01_run_all_simulations.py
   ```

   This script will place the simulated data into `data/sim_pet_data/<dataset>/data_tensors.pt`.
   You can change `countlevel = [1.0]`, to simulate data sets with different
   count levels. More counts means less noisy images.

3. **Reconstruct Simulated Data**

   Once the data is simulated, run the following script to perform listmode MLEM 
   reconstruction for all simulated datasets:
   ```bash
   python 02_run_all_mlem_recons.py
   ```

   This script will place the reconstructed images `data/sim_pet_data/<dataset>/mlem_reconstructions.pt`
   and also create a `mlem.png` showing MLEM images and ground truth images.

## Training a simple image to image denoising neural network without using data fidelity

Once the data is simulated and the MLEM reconstruction are generated,
we can train a simple neural network that maps from the noisy MLEM image to the
simulated ground truth in image space.

```bash
python 04_train_img_to_img_denoiser.py MiniConvNet --model_kwargs '{"num_features":16, "num_hidden_layers":4}' --num_epochs 500
```

or 

```bash
python 04_train_img_to_img_denoiser.py UNet3D --model_kwargs '{"features":[16,32]}' --num_epochs 500
```

The first argument must be the name of torch model defined in models.py that 
defines the image-to-image model architecture.
You can define your own custom model architectures, but you should not change / overwrite
existing classes to keep backward compatibility.
If you implement custom models, make sure that the output is non-negative
(e.g. by adding a final ReLU).

This script will save model checkpoints and a pdf containing evaluation metrics
in `checkpoints_denoiser`.

To evaluate a trained model (checkpoint) you can run
```bash
python 05_eval_img_to_img_denoiser.py checkpoint_denoiser/my_run/my_ckpt.pt
```

## Training an unrolled network using data fidelity gradient layers

Once we have pre-trained an image to image denoiser, we can train more
complicated unrolled network with blocks consisting of data fidelity gradient
layers follwed by a trainable neural network.

```bash
python 06_train_unrolled_net.py checkpoints_denoiser/MiniConvNet_1/best.pth --lr 1e-3 --num_epochs 100 --num_blocks 4
```

The first argument must be saved checkpoint path that contains the pre-trained denoiser.
Note that training these networks will take much more memory and time, since we
have to backpropagate through the data fidelity gradient layers (requires
2 forward and 1 back projection per layer).

This script will save model checkpoints and a pdf containing evaluation metrics
in `checkpoints_unrolled`.

To evaluate a trained model (checkpoint) you can run
```bash
python 07_eval_unrolled_net.py checkpoint_unrolled/my_run/my_ckpt.pt
```
