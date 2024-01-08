# LightMHC: A Light Model for pMHC Structure Prediction with Graph Neural Networks

[![Python Version](https://img.shields.io/badge/python-3.8-blue.svg)](https://docs.python.org/3.8/library/index.html)
[![pytorch](https://img.shields.io/badge/PyTorch-1.8.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![license](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)](LICENSE)

Welcome to LightMHC, a Python framework for predicting peptide-MHC (pMHC) structures using deep
learning. This repository houses the codebase for LightMHC, which is designed for class-I pMHC
prediction with peptide sequences ranging from 8 to 13 amino acids. The
[paper](https://www.biorxiv.org/content/10.1101/2023.11.21.568015v1) associated with this repository
was accepted at the Machine Learning for Structural Biology workshop at the NeurIPS 2023 conference.

## Requirements

This project has the following dependencies:

- Conda
- Python 3.8
- CUDA 11.1 to use a GPU
- PyRosetta license credentials

## Installation

1. Clone this repository to your local machine

2. Install the conda environment by running:

```bash
conda create --yes --name lightmhc_env python=3.8
conda env update --name lightmhc_env --file environment.yml
```

If you have a MacOS system or do not have a CUDA-enabled GPU on your laptop, you can install the
CPU-only version with:

```bash
conda create --yes --name lightmhc_env python=3.8
conda env update --name lightmhc_env --file environment_no_gpu.yml
```

3. Install PyRosetta, replacing the placeholders with your own credentials: Linux/Ubuntu operating
   system

```bash
wget https://USERNAME:PASSWORD@conda.graylab.jhu.edu/linux-64/pyrosetta-2021.34+release.5eb89ef-py38_0.tar.bz2
conda activate lightmhc_env
conda install pyrosetta-2021.34+release.5eb89ef-py38_0.tar.bz2
```

MacOS operating system

```bash
wget https://USERNAME:PASSWORD@conda.graylab.jhu.edu/osx-64/pyrosetta-2021.34+release.5eb89ef1fc1-py38_0.tar.bz2
conda activate lightmhc_env
conda install pyrosetta-2021.34+release.5eb89ef1fc1-py38_0.tar.bz2
```

4. Install LightMHC using pip:

```bash
pip install -e .
```

## Usage

To perform pMHC structure prediction using LightMHC, follow these steps:

1. Prepare a CSV file with the following columns:

- `pdb_id`: The PDB ID of the structure.
- `peptide`: The peptide sequence (8-13 amino acids).
- `mhc`: The MHC sequence (excluding signal peptide residues, e.g., starting at 'GSH').

2. Run the following command, specifying the input CSV path, output directory, and the number of CPU
   cores to use:

```bash
python inference.py data.input_csv_path=YOUR_INPUT_DIR data.output_dir=YOUR_OUTPUT_DIR model.n_cpus=YOUR_CPU_NUMBER
```

## Examples

We have provided an example input CSV file and the resulting structures in the `examples/`
directory.

## Acknowledgements

This project includes code adapted from Abanades et al., 2023.

```object
Abanades B, Wong WK, Boyles F, Georges G, Bujotzek A, Deane CM. ImmuneBuilder: Deep-Learning models for predicting the structures of immune proteins. Commun Biol. 2023 May 29.
```

## Citing this work

If you use LightMHC in your work, please cite it using:

```object
@article{delaunay2023lightmhc,
	author = {Antoine P Delaunay and Yunguan Fu and Nikolai Gorbushin and Robert McHardy and Bachir A Djermani and Liviu Copoiu and Michael Rooney and Maren Lang and Andrey Tovchigrechko and Ugur Sahin and Karim Beguir and Nicolas Lopez Carranza},
	title = {LightMHC: A Light Model for pMHC Structure Prediction with Graph Neural Networks},
	year = {2023},
	doi = {10.1101/2023.11.21.568015},
	publisher = {Cold Spring Harbor Laboratory},
	journal = {bioRxiv}
}
```

If you have any questions or feedback on the code and models, please feel free to reach out to us.

Thank you for your interest in our work!

## License

[LightMHC: A Light Model for pMHC Structure Prediction with Graph Neural Networks](https://github.com/instadeepai/lightmhc/) © 2023 by [InstaDeep Ltd](https://www.instadeep.com/) is licensed under [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1) <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1">
<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1">
<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1">

## Disclaimer of Warranties
We refer hereinbelow to the section 5 of the [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1) license.
```
  a. UNLESS OTHERWISE SEPARATELY UNDERTAKEN BY THE LICENSOR, TO THE
     EXTENT POSSIBLE, THE LICENSOR OFFERS THE LICENSED MATERIAL AS-IS
     AND AS-AVAILABLE, AND MAKES NO REPRESENTATIONS OR WARRANTIES OF
     ANY KIND CONCERNING THE LICENSED MATERIAL, WHETHER EXPRESS,
     IMPLIED, STATUTORY, OR OTHER. THIS INCLUDES, WITHOUT LIMITATION,
     WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR
     PURPOSE, NON-INFRINGEMENT, ABSENCE OF LATENT OR OTHER DEFECTS,
     ACCURACY, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT
     KNOWN OR DISCOVERABLE. WHERE DISCLAIMERS OF WARRANTIES ARE NOT
     ALLOWED IN FULL OR IN PART, THIS DISCLAIMER MAY NOT APPLY TO YOU.

  b. TO THE EXTENT POSSIBLE, IN NO EVENT WILL THE LICENSOR BE LIABLE
     TO YOU ON ANY LEGAL THEORY (INCLUDING, WITHOUT LIMITATION,
     NEGLIGENCE) OR OTHERWISE FOR ANY DIRECT, SPECIAL, INDIRECT,
     INCIDENTAL, CONSEQUENTIAL, PUNITIVE, EXEMPLARY, OR OTHER LOSSES,
     COSTS, EXPENSES, OR DAMAGES ARISING OUT OF THIS PUBLIC LICENSE OR
     USE OF THE LICENSED MATERIAL, EVEN IF THE LICENSOR HAS BEEN
     ADVISED OF THE POSSIBILITY OF SUCH LOSSES, COSTS, EXPENSES, OR
     DAMAGES. WHERE A LIMITATION OF LIABILITY IS NOT ALLOWED IN FULL OR
     IN PART, THIS LIMITATION MAY NOT APPLY TO YOU.

  c. The disclaimer of warranties and limitation of liability provided
     above shall be interpreted in a manner that, to the extent
     possible, most closely approximates an absolute disclaimer and
     waiver of all liability.
```
