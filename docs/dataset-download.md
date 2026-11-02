# Dataset Download

This document will discuss how to download and use the ablation dataset for the Splatify project. 

The dataset can be found at this location as a zip file for download, or it can be recreated using the `ablation_dataset_creation.ipynb` notebook. These produce the same final result with the same shared seed. 

The existing dataset can be found in the shared project google drive folder. 

When using the dataset, please unzip any datasets into the `data/` directory so shared utilities can easily be used and called. 

## Dataset Format

The dataset contains product images, matching 3D models, and two CSV files that connect them.

```text
data/abo/
├── images/<product_type>/<spin_id>/<azimuth>.jpg
├── models/<3dmodel_id>.glb
├── meshes/<3dmodel_id>.ply
├── image_manifest.csv
└── selected_products.csv
```

Each product has 72 ordered images. `image_manifest.csv` also marks the images used for the 36-, 18-, and 9-image experiments. `selected_products.csv` lists each product and its related image and model IDs.

The images are ordered by product type into 5 main classes under investigation with 20 products each CHAIR, DRESSER, LAMP, SOFA, VASE as they have distince features.

The corresponding mesh can be found in the meshes folder and correspondence can be checked using the `selected_products.csv` file. 

The models folder is unused but holds the base 3D models as .glb files.

## Using Data Loading Utils

After installing the dataset in `data/abo/`, use `ABODataset` from the project root:

Run the loader tests from the project root with:

```bash
pytest tests/test_ablation_dataset.py
```

For using the helper in your own code:

```python
from utils.ablation_dataset import ABODataset

dataset = ABODataset()
product = dataset.list_products(category="CHAIR")[0]
sample = dataset.get_product(product["spin_id"], num_images=18)

print(sample["image_paths"])
print(sample["mesh_path"])
```

`num_images` controls how many evenly spaced views are returned. The standard experiment sizes are 72, 36, 18, and 9. Use `load_images()` or `load_mesh()` when the files need to be loaded into memory.


The tests check the loader with sample data. If the downloaded dataset exists in `data/abo/`, they also verify its images and meshes.
