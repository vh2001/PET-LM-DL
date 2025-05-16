import subprocess
from pathlib import Path

data_dir = Path("data/brainweb_petmr_v2")
contrast_nums = [0, 1, 2]
countlevels = [1.0]
seeds = [1]

# Find all subject folders (e.g., subject01, subject02, ...)
subject_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("subject")])

for subject_dir in subject_dirs:
    subject_num = int(subject_dir.name.replace("subject", ""))
    for contrast_num in contrast_nums:
        for countlevel in countlevels:
            for seed in seeds:
                out_dir = Path(
                    f"data/sim_pet_data/subject{subject_num:02}_contrast_{contrast_num}_countlevel_{countlevel}_seed_{seed}"
                )
                data_tensor_file = out_dir / "data_tensors.pt"
                if data_tensor_file.exists():
                    print(f"Skipping subject {subject_num}, contrast {contrast_num}, countlevel {countlevel}, seed {seed}: data already exists.")
                    continue
                cmd = [
                    "python",
                    "01_simulate_data.py",
                    "--subject_num", str(subject_num),
                    "--contrast_num", str(contrast_num),
                    "--countlevel", str(countlevel),
                    "--seed", str(seed)
                ]
                print(f"Running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)