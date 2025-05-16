import subprocess
from pathlib import Path

sim_data_dir = Path("data/sim_pet_data")

# Find all simulation folders
sim_folders = sorted([d for d in sim_data_dir.iterdir() if d.is_dir()])

for sim_folder in sim_folders:
    recon_file = sim_folder / "mlem_reconstructions.pt"
    if recon_file.exists():
        print(f"Skipping {sim_folder}: reconstruction already exists.")
        continue
    cmd = ["python", "02_lm_mlem.py", str(sim_folder)]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
