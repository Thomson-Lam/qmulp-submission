# Experimental data

## HMEQ

The Home Equity dataset, or HMEQ, is available from this [CSV download](https://www.creditriskanalytics.net/uploads/1/9/5/1/19511601/hmeq.csv).
The download contains no citation or license notice. Confirm attribution and redistribution terms before public deposit.

`hmeq.csv` was downloaded from that URL for this port. SHA-256:
`dfdbc2b7cdf728a15b53e323cde6127995715dfa6b178bd3c1e3d9916d0367aa`.
It contains 5,960 rows and 13 columns:

- `BAD`: Loan outcome, `1` means defaulted or seriously delinquent and `0` means repaid.
- `LOAN`: Requested loan amount, in dollars.
- `MORTDUE`: Amount owed on the existing mortgage, in dollars.
- `VALUE`: Current property value, in dollars.
- `REASON`: Loan purpose. `DebtCon` means debt consolidation; `HomeImp` means home improvement.
- `JOB`: Borrower's occupational category.
- `YOJ`: Years at the current job.
- `DEROG`: Number of major derogatory credit reports.
- `DELINQ`: Number of delinquent credit lines.
- `CLAGE`: Age of the oldest credit line, in months.
- `NINQ`: Number of recent credit inquiries.
- `CLNO`: Number of credit lines.
- `DEBTINC`: Debt-to-income ratio, expressed as a percentage.

Stage 01 reads `data/hmeq.csv` and writes the combined `data/hmeq_processed.parquet` file.
It writes the frozen split and bootstrap indices under `results/outputs/`.
The older `hmeq_train_processed.parquet` and `hmeq_test_processed.parquet` files are not outputs or required inputs of the current pipeline.
See [cleaning.py](../src/hmeq/cleaning.py) and [the data stage](../src/stages/data.py).
Raw and processed HMEQ files are excluded from Git.

The software LICENSE notice grants no data rights.

## Freddie Mac

Obtain the author-supplied extracts under their Freddie Mac and author access terms. Do not reconstruct or replace the supplied targets.
Place `Quantum_florida_202303*.csv` and `Quantum_florida_202403*.csv` in `data/`.
The expected SHA-256 hashes recorded in the root README are:

- March 2023: `f9740a85ac7bed545908bbe069298bb9e07b43bcc75aba354d3f75ea5fda7927`
- March 2024: `dddab754e9d6bc2c167896268151e0b4aa4709e8f92ed5399925e4a79fcb5a86`

The notebook [01_fmac_cleaning.ipynb](../notebooks/01_fmac_cleaning.ipynb) prepares the cohorts and calls the data stage.

## Dataset-free experiments

The existing `scripts/run_synthetic.py` and `scripts/ec_counterexample.py` generate
their own experimental data. Usage: `bash scripts/run.sh -m scripts.<name>`.
