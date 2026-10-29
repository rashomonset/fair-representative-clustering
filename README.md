# Fair and Representative Clustering for Data Summarization

## Setup and Requirements
python >= 3.10
```bash
python -m pip install numpy==1.26.4 scipy==1.15.3 pandas==1.5.2 
    scikit-learn==1.2.0 gurobipy==13.0.1 ucimlrepo==0.0.7 tqdm psutil
```
## Layout

Directories `k_supplier/` and `k_means/` contain the following files: 

* `assignmentLP.py`: Assignment LP with Randomized Rounding.
* `profileILP.py`: Profile ILP.
* `runner.py`: Code for Algorithmic Fairness Evaluation on Real-World Datasets. 
* `data_utils.py` Code for Data Loading and Preprocessing. 
* `utils.py`: Helper Functions to compute the Utility Matrix, Metrics, Clustering Cost and more.

`k_supplier/benchmark.py`: Code for Benchmarking the Scalability of Algorithms using Synthetic Data. 

## Usage - Algorithmic Fairness Evaluation 

Run commands from either `k_supplier/` or `k_means/`:

```bash
mkdir -p logs results
python runner.py --dataset [dataset] --objective [global|local] 
```
- `--dataset`: `heart`, `student`, `credit_card`, `bank`, `census_adult`, or `movie_lens`.
- `--objective`: `global` or `local`.

MovieLens requires
`users.dat`, `ratings.dat`, and `movies.dat` inside `data/ml-1m/` at the root.

## Usage - Scalability 

Run from `k_supplier/`:

```bash
python benchmark.py --objective [global|local] --method [method] --output [csv_path]
```
- `--objective`: default `global`.
- `--method`: `assignmentLP`, `profileILP`, or `fairksupplier`; omit to run all three.
- `--output`: default `results/benchmark.csv`

## Baselines from Prior Work on Fair Clustering

The k-supplier baselines are adapted from [fair-k-supplier-source](https://github.com/suhastheju/fair-k-supplier-source).

The k-means baselines are adapted from [diversity-aware-clustering](https://github.com/suhastheju/diversity-aware-clustering). 
The repository is expected to be stored in `k_means/divkmeans/` and requires older versions of numpy. 
Run using:
```bash
python runner_divkmeans.py --dataset [dataset] --split [equitable|proportional] 
```
