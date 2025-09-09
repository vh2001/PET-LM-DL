import json
import torch
import parallelproj
from pathlib import Path
from array_api_compat import device


def load_lm_pet_data(odir: Path, skip_raw_data: bool = False):
    """
    Loads PET listmode data and returns the linear operator, contamination list, and adjoint ones.
    """

    if skip_raw_data:
        return torch.load(odir / "ground_truth.pt")
    else:
        data_tensors = torch.load(odir / "data_tensors.pt")
        ground_truth = torch.load(odir / "ground_truth.pt")
        x_true = ground_truth
        event_start_coords = data_tensors["event_start_coords"]
        event_end_coords = data_tensors["event_end_coords"]
        event_tofbins = data_tensors["event_tofbins"]
        att_list = data_tensors["att_list"]
        contamination_list = data_tensors["contamination_list"]
        adjoint_ones = data_tensors["adjoint_ones"]

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
            in_shape,
            sigma=fwhm_data_mm
            / (2.35 * torch.tensor(voxel_size, dtype=torch.float32).to(device(x_true))),
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

    def __init__(
        self, data_dirs: list[Path], shuffle: bool = True, skip_raw_data: bool = False, use_early_mlem: bool = True
    ):
        self._data_dirs = data_dirs.copy()
        if shuffle:
            idx = torch.randperm(len(self._data_dirs))
            self._data_dirs = [self._data_dirs[i] for i in idx]
        self._skip_raw_data = skip_raw_data
        self._use_early_mlem = use_early_mlem

    @property
    def data_dirs(self):
        return self._data_dirs

    def __len__(self):
        return len(self._data_dirs)

    def __getitem__(self, idx):
        odir = self._data_dirs[idx]
        if self._use_early_mlem:
            input_img = torch.load(odir / "mlem_reconstructions.pt")[
                "x_mlem_early_filtered"
            ]
        else:
            input_img = torch.load(odir / "mlem_reconstructions.pt")[
                "x_mlem_filtered"
            ]

        if self._skip_raw_data:
            return {
                "input": input_img.unsqueeze(0),
                "target": load_lm_pet_data(odir, skip_raw_data=True).unsqueeze(0),
            }
        else:
            lm_pet_lin_op, contamination_list, adjoint_ones, x_true = load_lm_pet_data(
                odir
            )
            return {
                "input": input_img,
                "target": x_true,
                "lm_pet_lin_op": lm_pet_lin_op,
                "contamination_list": contamination_list,
                "adjoint_ones": adjoint_ones,
            }


def brainweb_collate_fn(batch):
    # batch is a list of dicts
    return {
        "input": torch.stack([item["input"].unsqueeze(0) for item in batch]),
        "target": torch.stack([item["target"].unsqueeze(0) for item in batch]),
        "lm_pet_lin_ops": [item["lm_pet_lin_op"] for item in batch],
        "contamination_lists": [item["contamination_list"] for item in batch],
        "adjoint_ones": torch.stack([item["adjoint_ones"] for item in batch]),
        "diag_preconds": torch.stack([
            item["input"] / item["adjoint_ones"] + 1e-6 for item in batch
        ]),
    }


if __name__ == "__main__":
    from utils import LMNegPoissonLogLGradientLayer

    torch.manual_seed(42)

    skip_raw = True

    dataset = BrainwebLMPETDataset(
        sorted(list(Path("data/sim_pet_data").glob("subject*"))), skip_raw_data=skip_raw
    )

    if skip_raw:
        collate_fn = None
    else:
        collate_fn = brainweb_collate_fn

    loader = torch.utils.data.DataLoader(dataset, batch_size=3, collate_fn=collate_fn)

    batch = next(iter(loader))

    if not skip_raw:
        lm_logL_grad_layer = LMNegPoissonLogLGradientLayer.apply
        logL_grads = lm_logL_grad_layer(
            batch["input"],
            batch["lm_pet_lin_ops"],
            batch["contamination_lists"],
            batch["adjoint_ones"],
            batch["diag_preconds"],
        )
