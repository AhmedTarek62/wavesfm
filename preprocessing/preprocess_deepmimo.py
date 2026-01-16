"""Build a DeepMIMO cache with LoS/NLoS and beam prediction labels."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import DeepMIMOv3


SCENARIO_NAMES = (
    "city_18_denver",
    "city_15_indianapolis",
    "city_19_oklahoma",
    "city_12_fortworth",
    "city_11_santaclara",
    "city_7_sandiego",
)


def get_parameters(scenario: str, dataset_folder: str) -> tuple[dict, dict, int, int, int]:
    n_ant_bs = 32
    n_ant_ue = 1
    n_subcarriers = 32
    scs = 30e3

    row_column_users = {
        "city_18_denver": {"n_rows": 85, "n_per_row": 82},
        "city_15_indianapolis": {"n_rows": 80, "n_per_row": 79},
        "city_19_oklahoma": {"n_rows": 82, "n_per_row": 75},
        "city_12_fortworth": {"n_rows": 86, "n_per_row": 72},
        "city_11_santaclara": {"n_rows": 47, "n_per_row": 114},
        "city_7_sandiego": {"n_rows": 71, "n_per_row": 83},
    }

    parameters = DeepMIMOv3.default_params()
    parameters["dataset_folder"] = dataset_folder
    parameters["scenario"] = scenario

    if scenario == "O1_3p5":
        parameters["active_BS"] = np.array([4])
    elif scenario in ["city_18_denver", "city_15_indianapolis"]:
        parameters["active_BS"] = np.array([3])
    else:
        parameters["active_BS"] = np.array([1])

    if scenario == "Boston5G_3p5":
        parameters["user_rows"] = np.arange(
            row_column_users[scenario]["n_rows"][0],
            row_column_users[scenario]["n_rows"][1],
        )
    else:
        parameters["user_rows"] = np.arange(row_column_users[scenario]["n_rows"])

    parameters["bs_antenna"]["shape"] = np.array([n_ant_bs, 1])
    parameters["bs_antenna"]["rotation"] = np.array([0, 0, -135])
    parameters["ue_antenna"]["shape"] = np.array([n_ant_ue, 1])
    parameters["enable_BS2BS"] = False
    parameters["OFDM"]["subcarriers"] = n_subcarriers
    parameters["OFDM"]["selected_subcarriers"] = np.arange(n_subcarriers)
    parameters["OFDM"]["bandwidth"] = scs * n_subcarriers / 1e9
    parameters["num_paths"] = 20

    return parameters, row_column_users, n_ant_bs, n_ant_ue, n_subcarriers


def uniform_sampling(dataset: list[dict], sampling_div: list[int], n_rows: int, users_per_row: int) -> np.ndarray:
    cols = np.arange(users_per_row, step=sampling_div[0])
    rows = np.arange(n_rows, step=sampling_div[1])
    return np.array([j + i * users_per_row for i in rows for j in cols])


def select_by_idx(dataset: list[dict], idxs: np.ndarray) -> list[dict]:
    dataset_t = []
    for bs_idx in range(len(dataset)):
        dataset_t.append({})
        for key in dataset[bs_idx].keys():
            dataset_t[bs_idx]["location"] = dataset[bs_idx]["location"]
            dataset_t[bs_idx]["user"] = {k: dataset[bs_idx]["user"][k][idxs] for k in dataset[bs_idx]["user"]}
    return dataset_t


def deepmimo_data_gen(scenario: str, dataset_folder: str) -> dict:
    parameters, row_column_users, _, _, _ = get_parameters(scenario, dataset_folder)
    deepmimo_dataset = DeepMIMOv3.generate_data(parameters)
    uniform_idxs = uniform_sampling(
        deepmimo_dataset,
        [1, 1],
        len(parameters["user_rows"]),
        users_per_row=row_column_users[scenario]["n_per_row"],
    )
    return select_by_idx(deepmimo_dataset, uniform_idxs)[0]


def deepmimo_data_cleaning(data: dict) -> tuple[np.ndarray, np.ndarray]:
    idxs = np.where(data["user"]["LoS"] != -1)[0]
    cleaned = np.asarray(data["user"]["channel"][idxs]) * 1e6
    return cleaned, idxs


def steering_vec(array: np.ndarray, phi: float = 0, theta: float = 0, kd: float = np.pi) -> np.ndarray:
    idxs = DeepMIMOv3.ant_indices(array)
    resp = DeepMIMOv3.array_response(idxs, phi, theta + np.pi / 2, kd)
    return resp / np.linalg.norm(resp)


def beam_labels(data: dict, scenario: str, n_beams: int, dataset_folder: str) -> np.ndarray:
    parameters, _row_column_users = get_parameters(scenario, dataset_folder)[:2]
    n_users = len(data["user"]["channel"])
    n_subbands = 1
    fov = 180

    beam_angles = np.around(np.arange(-fov / 2, fov / 2 + 0.1, fov / (n_beams - 1)), 2)
    f_mat = np.array([
        steering_vec(
            parameters["bs_antenna"]["shape"],
            phi=azi * np.pi / 180,
            kd=2 * np.pi * parameters["bs_antenna"]["spacing"],
        ).squeeze()
        for azi in beam_angles
    ])

    full_dbm = np.zeros((n_beams, n_subbands, n_users), dtype=float)
    for ue_idx in tqdm(range(n_users), desc=f"Beam labels ({scenario})"):
        if data["user"]["LoS"][ue_idx] == -1:
            full_dbm[:, :, ue_idx] = np.nan
        else:
            chs = f_mat @ data["user"]["channel"][ue_idx]
            full_linear = np.abs(np.mean(chs.squeeze().reshape((n_beams, n_subbands, -1)), axis=-1))
            full_dbm[:, :, ue_idx] = np.around(20 * np.log10(full_linear) + 30, 1)

    best_beams = np.argmax(np.mean(full_dbm, axis=1), axis=0).astype(float)
    best_beams[np.isnan(full_dbm[0, 0, :])] = np.nan
    return best_beams


def _channels_to_ri(channels: np.ndarray) -> np.ndarray:
    arr = np.asarray(channels)[:, 0]
    if np.iscomplexobj(arr):
        return np.stack((arr.real, arr.imag), axis=1).astype(np.float32)
    return arr.astype(np.float32)


def preprocess_deepmimo(
    output: Path,
    scenarios: Iterable[str],
    dataset_folder: str,
    n_beams: int = 64,
    compression: str | None = None,
    overwrite: bool = False,
) -> Path:
    output = Path(output)
    scenarios = list(scenarios)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    samples: List[np.ndarray] = []
    los_labels: List[np.ndarray] = []
    beam_labels_list: List[np.ndarray] = []
    scenario_labels: List[np.ndarray] = []

    for scenario in scenarios:
        data = deepmimo_data_gen(scenario, dataset_folder)
        cleaned, idxs = deepmimo_data_cleaning(data)
        los = data["user"]["LoS"][idxs].astype(np.int64)
        beams = beam_labels(data, scenario, n_beams, dataset_folder)[idxs].astype(np.int64)
        samples.append(_channels_to_ri(cleaned))
        los_labels.append(los)
        beam_labels_list.append(beams)
        scenario_labels.append(np.asarray([scenario] * len(los), dtype=object))

    sample_arr = np.concatenate(samples, axis=0)
    los_arr = np.concatenate(los_labels, axis=0)
    beam_arr = np.concatenate(beam_labels_list, axis=0)
    scenario_arr = np.concatenate(scenario_labels, axis=0)

    # Resize to 224x224 and compute channel-wise mean/std on resized data.
    resized = F.interpolate(
        torch.from_numpy(sample_arr),
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
    ).numpy()
    mean = resized.mean(axis=(0, 2, 3), dtype=np.float64)
    std = resized.std(axis=(0, 2, 3), dtype=np.float64)
    std = np.clip(std, 1e-12, None)
    sample_arr = ((resized - mean[None, :, None, None]) / std[None, :, None, None]).astype(np.float32)

    n = sample_arr.shape[0]
    sample_shape = sample_arr.shape[1:]
    chunk = min(1024, n) if n > 0 else 1
    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output, "w") as h5:
        h5.create_dataset(
            "sample",
            shape=(n, *sample_shape),
            dtype="float32",
            chunks=(chunk, *sample_shape),
            compression=compression,
        )
        h5.create_dataset("label_los", shape=(n,), dtype="int64", chunks=(chunk,), compression=compression)
        h5.create_dataset("label_beam", shape=(n,), dtype="int64", chunks=(chunk,), compression=compression)
        h5.create_dataset("scenario", shape=(n,), dtype=str_dtype, chunks=(chunk,), compression=compression)

        h5["sample"][:] = sample_arr
        h5["label_los"][:] = los_arr
        h5["label_beam"][:] = beam_arr
        h5["scenario"][:] = scenario_arr

        h5.attrs["scenarios"] = json.dumps(list(scenarios))
        h5.attrs["n_beams"] = int(n_beams)
        h5.attrs["sample_shape"] = json.dumps(list(sample_shape))
        h5.attrs["sample_format"] = "ri"
        h5.attrs["scale_factor"] = 1e6
        h5.attrs["mean"] = json.dumps([float(x) for x in mean])
        h5.attrs["std"] = json.dumps([float(x) for x in std])
        h5.attrs["labels_los"] = json.dumps(["NLoS", "LoS"])
        h5.attrs["version"] = "v1"
        h5.attrs["dataset_folder"] = str(dataset_folder)

    return output


def _parse_csv_list(raw: str, cast=str) -> list:
    return [cast(item) for item in raw.split(",") if item.strip() != ""]


def _clone_repo(url: str, dest: Path) -> None:
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Clone destination already exists and is not empty: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", url, str(dest)], check=True)
    subprocess.run(["git", "lfs", "pull"], check=True, cwd=str(dest))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute DeepMIMO channels with LoS and beam labels.")
    p.add_argument("--output", required=True, help="Output path for the h5 cache.")
    p.add_argument(
        "--scenarios",
        default=None,
        help="Comma-separated scenario names (default: all supported).",
    )
    p.add_argument(
        "--scenario-idxs",
        default=None,
        help="Comma-separated indices into the default scenario list.",
    )
    p.add_argument(
        "--dataset-folder",
        default="./scenarios",
        help="Path to DeepMIMO scenario folder (default: ./scenarios).",
    )
    p.add_argument(
        "--clone-scenarios",
        action="store_true",
        help="Clone the DeepMIMO scenarios repo into --dataset-folder, then remove it after preprocessing.",
    )
    p.add_argument(
        "--clone-url",
        default="https://huggingface.co/datasets/wi-lab/lwm",
        help="Repo URL to clone when using --clone-scenarios.",
    )
    p.add_argument("--n-beams", type=int, default=64, help="Number of beams for prediction labels.")
    p.add_argument(
        "--compression",
        default="none",
        choices=["gzip", "lzf", "none"],
        help="h5 dataset compression (default: none).",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.scenarios:
        scenario_list = _parse_csv_list(args.scenarios, str)
    elif args.scenario_idxs:
        idxs = _parse_csv_list(args.scenario_idxs, int)
        scenario_list = [SCENARIO_NAMES[i] for i in idxs]
    else:
        scenario_list = list(SCENARIO_NAMES)

    comp = None if args.compression == "none" else args.compression
    dataset_folder = Path(args.dataset_folder)
    cloned = False
    try:
        if args.clone_scenarios:
            _clone_repo(args.clone_url, dataset_folder)
            cloned = True
        out = preprocess_deepmimo(
            output=Path(args.output),
            scenarios=scenario_list,
            dataset_folder=str(dataset_folder),
            n_beams=args.n_beams,
            compression=comp,
            overwrite=args.overwrite,
        )
    finally:
        if cloned and dataset_folder.exists():
            shutil.rmtree(dataset_folder)
    print(f"Wrote DeepMIMO cache to {out}")


if __name__ == "__main__":
    main()
