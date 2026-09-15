# Data is intentionally not committed

`data/raw/` holds official MaleCNS source files and `data/full/` holds derived
artifacts for the full 166,700-neuron service. Both are excluded from Git.

Download the three official files listed in the project README into `data/raw/`,
then run `python scripts/prepare_full_data.py`.
