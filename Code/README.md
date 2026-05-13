# CodaST

## Installation

**1. Create Environment**
```bash
conda create -n CodaST python=3.8
conda activate CodaST
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Install R packages** (for mclust)
```R
install.packages("mclust")
```

## Project Structure

```
.
├── Code/                      # Source code
│   ├── GAT.py                 # Graph Attention Network layers
│   ├── model.py               # Core model architecture
│   ├── network.py             # Training pipeline
│   ├── layers.py              # Additional neural network layers
│   ├── preprocess.py          # Data preprocessing utilities
│   └── utils.py               # Helper functions
├── data/                      # Sample datasets
├── results/                   # Output results
├── run.py                     # Main training script
└── requirements.txt           # Python dependencies
```

## Usage

```python
import torch
import scanpy as sc
from CodaST import network as CodaST
from CodaST.utils import clustering

# Load data
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
adata = sc.read_visium(file_path, count_file='filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()

# Train model
model = CodaST.CodaST(adata, device=device, epochs=600)
adata = model.train()

# Clustering
clustering(adata, n_clusters=7, radius=35, method='mclust', refinement=True)
```

To run the full pipeline:
```bash
python run.py
```

## Troubleshooting

- **R Integration**: Ensure R is installed and the `R_HOME` environment variable is set.
- **GPU Memory**: Reduce batch size or use CPU mode for large datasets.
- **Dependencies**: Re-run `pip install -r requirements.txt` if you encounter missing packages.
