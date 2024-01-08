"""Run inference on GNN structure models."""
import os
import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import Any, List

import hydra
import numpy as np
import omegaconf
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from lightmhc.core import create_dataset, get_dataloader, get_model, get_transforms
from lightmhc.predict_structure import dump_pdb, fix_structures


def inference_step(
    model: nn.Module,
    data_test: List[Any],
    device: torch.device,
) -> np.ndarray:
    """Run one inference step on a batch of data.

    Args:
        model: PyTorch model to use.
        data_test: A batch of data for one training step.
        device: Device to store tensors.

    Returns:
        Predictions on this batch. Shape = (batch, n_layers, n_aa, 14, 3).
    """
    with torch.no_grad():
        data_test = data_test.to(device)
        preds, *_ = model(data_test)
        preds = preds.cpu().numpy()
        # Shape: (n_layers, batch, n_aa, 14, 3) --> (batch, n_layers, n_aa, 14, 3).
        preds = np.transpose(preds, axes=(1, 0, 2, 3, 4))
        preds = preds[:, -1]
    return preds


def workflow(
    conf: omegaconf.DictConfig,
    input_csv_path: Path,
    output_dir: Path,
    device: torch.device,
    workflow_index: int = 0,
    fix_pdb: bool = True,
) -> None:
    """For a given set of sequences, run the DL model, dump the predicted coordinates into a PDB and correct the PDB.

    Args:
        conf: Dictionary with configuration.
        input_csv_path: Path to csv containing input sequences.
        output_dir: Root output directory where predicted PDBs are saved.
        device: Device on which data is loaded and DL model run.
        workflow_index: Index of the workflow to avoid overlap in the output directories.
        fix_pdb: If true, apply Rosetta IdealizeMover to fix PDB bond lengths/angles. Default True.
    """
    dataset = create_dataset(
        input_csv_path, conf.data.chain_lengths, data_type=conf.data.data_type
    )
    transformations = get_transforms(conf)
    dataset.transform = transformations
    dataloader = get_dataloader(dataset, "test", conf)
    conf.model.max_seq_len = dataset.total_length
    first_chain, second_chain = sorted(dataset.chain_lengths.keys())[:2]

    dataset.max_chain1_len = dataset.chain_lengths[first_chain]
    dataset.max_chain2_len = dataset.chain_lengths[second_chain]

    # Create network, optimizer and loss
    model = get_model(conf.model)
    model = model.to(device)
    model.load_state_dict(torch.load(conf.model.checkpoint_path, map_location=device)["state_dict"])

    preds = []
    indexes: List[int] = []
    
    for index, data_test in zip(dataloader.batch_sampler, dataloader):
        preds_batch = inference_step(model, data_test, device)
        preds.extend(preds_batch)
        indexes.extend(index)

    dataset.transform = None
    ordered_dataset = dataloader.dataset[indexes]
    pdb_ids = ordered_dataset.pdb_id
    sequences = ordered_dataset.sequences
    sequences_len = ordered_dataset.sequences_len
    numbers = ordered_dataset.numbers

    preds = np.stack(preds)

    workflow_dir = output_dir / f"workflow_{workflow_index}"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    dump_pdb(
        workflow_dir,
        list(dataloader.dataset.chain_lengths.keys()),
        torch.from_numpy(preds),
        pdb_ids,
        sequences,
        sequences_len,
        numbers,
    )

    if fix_pdb:
        fix_structures(workflow_dir, workflow_dir)

    # Move PDBs from workflow-specific dir to output dir for future uploading to bucket.
    for file in workflow_dir.iterdir():
        file.rename(output_dir / file.name)

    # Remove workflow-specific dir.
    shutil.rmtree(workflow_dir)


def partition_csv(input_csv_path: Path, n_cpus: int = 1) -> List[Path]:
    """Split the dataset evenly across the CPUs.

    The input CSV is split into subset CSV for each cpu.

    Args:
        input_csv_path: Path of input CSV containing all the sequences.
        n_cpus: Number of CPUs on which to split the dataset.

    Returns:
        List of partitioned CSV paths.
    """
    df = pd.read_csv(input_csv_path)

    input_partitions = [df.iloc[i::n_cpus] for i in range(n_cpus)]
    partition_list = []

    for i, rows in enumerate(input_partitions):
        partition_path = input_csv_path.parent / f"{input_csv_path.stem}_cpu_{i}.csv"
        if rows.shape[0] > 0:
            rows.to_csv(partition_path, index=False)
            partition_list.append(partition_path)

    return partition_list


@hydra.main(config_path="conf/", config_name="config")
def main(args: omegaconf.DictConfig) -> None:
    """Main function to run over the sequences contained in an input csv.

    Args:
        args: Configuration dictionary.
    """
    OmegaConf.set_struct(args, False)

    if args.model.checkpoint_path == "":
        args.model.checkpoint_path = str(
            Path(os.path.dirname(os.path.realpath(__file__))).parent
            / "checkpoints"
            / "checkpoint_random.pt"
        )

    device = (
        torch.device("cuda:0")
        if (args.model.use_gpu and torch.cuda.is_available())
        else torch.device("cpu")
    )

    partition_list = partition_csv(Path(args.data.input_csv_path), args.model.n_cpus)

    if "fix_pdb" not in args.keys():
        args.fix_pdb = True

    workflow_args = [
        (
            args,
            Path(input_path),
            Path(args.data.output_dir),
            device,
            i,
            args.fix_pdb,
        )
        for i, input_path in enumerate(partition_list)
    ]
    with Pool(processes=args.model.n_cpus) as pool:
        pool.starmap(workflow, workflow_args)


# python inference.py  data.input_csv_path=YOUR_INPUT_DIR data.output_dir=YOUR_OUTPUT_DIR model.n_cpus=YOUR_CPU_NUMBER
if __name__ == "__main__":
    main()
