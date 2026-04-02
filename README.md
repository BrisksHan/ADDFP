# ADDFP

This repository contains the implementation of ADDFP, the Asymmetric Dual-task framework for Drug-side effect Frequency Prediction.

## Published paper

The paper based on this code has been published:

[Drug-side effect frequency prediction using an asymmetric multi-task learning approach](https://link.springer.com/article/10.1007/s44163-025-00616-y)

Journal: *Discover Artificial Intelligence*  
Published: November 28, 2025  
DOI: `10.1007/s44163-025-00616-y`

Copyable BibTeX:

```bibtex
@article{zhang2025drug,
  title = {Drug-side effect frequency prediction using an asymmetric multi-task learning approach},
  author = {Zhang, Han and Zhang, Zhan and Xiong, Jing and Jiao, Qingju and Guo, An and Yu, Yang and Sun, Hua},
  journal = {Discover Artificial Intelligence},
  volume = {5},
  number = {1},
  pages = {363},
  year = {2025},
  publisher = {Springer}
}
```

## Repository structure

- `src/`: model, training, evaluation, and data-processing code for the current ADDFP run path
- `data/`: prepared data files used by the current code
- `data.zip`: archive copy of the dataset; usually not needed if `data/` is already present
- `src/llm_inference.py`: legacy helper code for generating embeddings; not required for the bundled reproduction path

## Environment

No pinned environment file is included in this repository. For the current bundled reproduction path, the main dependencies are:

- Python 3
- PyTorch
- PyTorch Geometric
- RDKit
- NumPy
- SciPy
- pandas
- scikit-learn
- NetworkX
- openpyxl
- tqdm

`Transformers` is not required for the default ADDFP reproduction path because the repository now uses precomputed description embeddings stored in `data/`.

A CUDA GPU is recommended because the default device in `src/main.py` is `cuda:0`. CPU execution is possible by passing `--device cpu`, but it will be much slower.

## Data and model preparation

The repository already contains the files needed for the current `python3 src/main.py` workflow.

In normal use, no manual data preparation is required.

Files used by the current run path include:

- `data/drug_side_frequencies.pkl`
- `data/Text_similarity_five.pkl`
- `data/drug_SMILES.xlsx`
- `data/side_effect_label_750.pkl`
- `data/drug_description_embeddings.pkl`
- `data/side_description_embeddings.pkl`
- `data/data_cvs1/*`
- `data/data_cvs2/*`

Not required anymore for the default ADDFP reproduction path:

- `data/drug_target.xz`
- local `llm/` model folders
- `Transformers`
- manual creation of `results/` and `trained_model/`
- regenerating the drug and side-effect description embeddings before running

Included in the repository but not used by the current main run path:

- `data/drug_description.xlsx`
- `data/side_description.xlsx`
- `data/drug_id.xlsx`
- `data/effect_side_semantic.pkl`
- `data/glove_wordEmbedding.pkl`
- `data/independent_ds.pkl`
- `data/side_effect_description_embedding.pkl`

If your local copy is missing the extracted `data/` directory, you can restore it from the archive:

```bash
unzip data.zip
```

The current code reads the bundled precomputed `384`-dimensional description embeddings from:

- `data/drug_description_embeddings.pkl`
- `data/side_description_embeddings.pkl`

Output folders such as `results/` and `trained_model/` are created automatically when you run the script.

## Running the code

Run the default experiment from the repository root:

```bash
python3 src/main.py
```

This uses the current defaults in `src/main.py`:

- warm-start setting (`--cvs 1`)
- a single split (default fold 0)
- 30 epochs
- device `cuda:0`

Run the cold-start setting:

```bash
python3 src/main.py --cvs 2 --device cuda:0
```

Run all 10 folds and save the average result:

```bash
python3 src/main.py --run_all 0 --cvs 1 --epoch 30 --device cuda:0
```

Note: in the current implementation, `--run_all 0` runs all 10 folds, while the default `--run_all 1` runs a single split.

## Output files

- trained models: `trained_model/model_st_q5_fold_<split>.pt`
- per-split results: `results/cvs_<cvs>_result_st_q5_split_e<epoch>_<split>.txt`
- averaged 10-fold results: `results/average_cvs<cvs>_e<epoch>_result_st_q5_all.txt`

The reported metrics include `scc`, `rmse`, `mae`, `auroc`, and `aupr`.

## Citation

If you use this repository, please cite:

`Drug-side effect frequency prediction using an asymmetric multi-task learning approach`, *Discover Artificial Intelligence*, 2025.  
Link: https://link.springer.com/article/10.1007/s44163-025-00616-y
