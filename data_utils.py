import json
import torch
import parallelproj
from pathlib import Path


def load_lm_pet_data(odir: Path):
    """
    Loads PET listmode data and returns the linear operator, contamination list, and adjoint ones.
    """
    data_tensors = torch.load(odir / "data_tensors.pt")

    event_start_coords = data_tensors["event_start_coords"]
    event_end_coords = data_tensors["event_end_coords"]
    event_tofbins = data_tensors["event_tofbins"]
    att_list = data_tensors["att_list"]
    contamination_list = data_tensors["contamination_list"]
    adjoint_ones = data_tensors["adjoint_ones"]
    x_true = data_tensors["x_true"]

    with open(odir / "projector_parameters.json", "r", encoding="UTF8") as f:
        projector_parameters = json.load(f)

    in_shape = tuple(projector_parameters["in_shape"])
    voxel_size = tuple(projector_parameters["voxel_size"])
    img_origin = tuple(projector_parameters["img_origin"])
    fwhm_data_mm = projector_parameters["fwhm_data_mm"]
    tof_parameters = parallelproj.TOFParameters(
        **projector_parameters["tof_parameters"]
    )

    lm_proj = parallelproj.ListmodePETProjector(
        event_start_coords,
        event_end_coords,
        in_shape,
        voxel_size,
        img_origin,
    )

    # enable TOF in the LM projector
    lm_proj.tof_parameters = tof_parameters
    lm_proj.event_tofbins = event_tofbins
    lm_proj.tof = True

    lm_att_op = parallelproj.ElementwiseMultiplicationOperator(att_list)

    res_model = parallelproj.GaussianFilterOperator(
        in_shape, sigma=fwhm_data_mm / (2.35 * torch.tensor(voxel_size))
    )

    lm_pet_lin_op = parallelproj.CompositeLinearOperator(
        (lm_att_op, lm_proj, res_model)
    )

    return lm_pet_lin_op, contamination_list, adjoint_ones, x_true


class BrainwebLMPETDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for supervised PET learning.
    Loads data from disk on-the-fly to avoid high memory usage.
    Each sample returns:
      - input image (ones)
      - label image (x_true)
      - lm_pet_lin_op
      - contamination_list
      - adjoint_ones
    """

    def __init__(self, data_dirs: list[Path]):
        self.data_dirs = data_dirs

    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, idx):
        odir = self.data_dirs[idx]
        lm_pet_lin_op, contamination_list, adjoint_ones, x_true = load_lm_pet_data(odir)
        input_img = torch.load(odir / "mlem_reconstructions.pt")[
            "x_mlem_early_filtered"
        ]
        return {
            "input": input_img,
            "target": x_true,
            "lm_pet_lin_op": lm_pet_lin_op,
            "contamination_list": contamination_list,
            "adjoint_ones": adjoint_ones,
        }


if __name__ == "__main__":
    dataset = BrainwebLMPETDataset(
        sorted(list(Path("data/sim_pet_data").glob("subject*")))
    )
    d0 = dataset[0]
