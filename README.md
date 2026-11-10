# Splatify

Splatify is a 3D object reconstruction project focused on generating 3D models from images or smartphone video for use in e-commerce applications.

The project compares traditional photogrammetry, Gaussian Splatting, and pretrained learning-based reconstruction techniques to evaluate reconstruction accuracy, computational cost, and image requirements.

## Getting Started

Clone the repository and install the required Python dependencies.

```bash
git clone <repository-url>
cd splatify_project

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Loading Data

The project uses a 100-product subset of the Amazon Berkeley Objects (ABO) dataset, containing 72 images and a ground-truth 3D model for each product.

The dataset can be downloaded from the shared project Google Drive or recreated using `dataset-creation/ablation_dataset_creation.ipynb`.

Extract the dataset into `data/` before running the reconstruction pipelines.

The shared data loader is located at `utils/ablation_dataset.py`.

```python
from utils.ablation_dataset import ABODataset

dataset = ABODataset()

product = dataset.list_products(category="CHAIR")[0]
sample = dataset.get_product(product["spin_id"], num_images=18)

images = sample["image_paths"]
mesh = dataset.load_mesh(product["spin_id"])
```

The loader supports 72, 36, 18, and 9 evenly spaced images for ablation experiments.

For additional information, see [Dataset Download and Usage](docs/dataset-download.md).

Dataset Drive Link: https://drive.google.com/drive/folders/1qRc1bPHfJG5Cme-n0WHa9R7zfaYO0Fpb?usp=drive_link
Dataset Kaggle Link: https://kaggle.com/datasets/3d483d7c36f34e551cbfa717644544381bb03f513394a355cfa62dfb147a9f08

To verify the dataset loader:

```bash
pytest tests/test_ablation_dataset.py
```

## Metrics

Reconstruction techniques will be evaluated based on geometric accuracy, visual fidelity, computational requirements, and reconstruction reliability.

Primary geometric metrics include Chamfer Distance, F-score, and dimensional error, comparing reconstructed meshes against the ground-truth geometry.

For metric definitions and evaluation methods, see [Metrics](docs/metrics.md).
