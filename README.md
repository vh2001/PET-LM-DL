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

4. **Note**

   You can also run `01_simulate_data.py` and `02_lm_mlem.py` to simulate and
   reconstruct individual data sets.

## Supervised Training of a listmode PET network 

- `04_train_network.py` is a simple script for training a mini unrolled network on the simulated PET data.
It should be seen as a proof-of-concept script, that can be used to setup, train and optimize
more advanced architectures. 
