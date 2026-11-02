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

The `utils/ablation_dataset.py` file provides a shared interface for accessing the dataset. It allows team members to select products, load evenly spaced images, and access the corresponding ground-truth 3D meshes without manually navigating the dataset directories.

The loader automatically locates `data/abo/` relative to the repository root.

### Initializing the Dataset

Import the `ABODataset` class and initialize the dataset:

```python
from utils.ablation_dataset import ABODataset

dataset = ABODataset()
```

Products can be accessed using either their `spin_id` or `item_id`. To retrieve the available products, optionally filtered by category:

```python
products = dataset.list_products(category="CHAIR")
product = products[0]

spin_id = product["spin_id"]
```

### Selecting Images for Reconstruction

The loader supports any number of images between 1 and 72, selecting approximately equidistant views around the object.

The standard ablation configurations are 72, 36, 18, and 9 images, corresponding to every 1st, 2nd, 4th, and 8th image.

```python
# Get image paths without loading images into memory
images = dataset.get_image_paths(spin_id, num_images=18)

# Load images as PIL RGB objects
images = dataset.load_images(spin_id, num_images=18)
```

The images are returned in rotation order. The original files are not modified or duplicated.

### Loading Ground-Truth Meshes

Every selected product has an associated ground-truth 3D model, identified through `selected_products.csv`.

```python
# Get the corresponding mesh file path
mesh_path = dataset.get_mesh_path(spin_id)

# Load mesh geometry using trimesh
mesh = dataset.load_mesh(spin_id)

print(mesh.vertices.shape)
print(mesh.faces.shape)
```

By default, the loader uses the converted `.ply` mesh. The original `.glb` file can also be accessed using `mesh_format="glb"`.

```python
mesh = dataset.load_mesh(spin_id, mesh_format="glb")
```

The returned mesh is a `trimesh.Trimesh` object, allowing direct access to vertices, faces, surface area, and other geometric properties for reconstruction accuracy analysis.

### Loading a Complete Product

For convenience, `get_product()` returns the selected image paths, image indices, mesh path, and product metadata together.

```python
sample = dataset.get_product(spin_id, num_images=18)

images = sample["image_paths"]
mesh_path = sample["mesh_path"]
indices = sample["image_indices"]

print(sample["product_type"])
print(indices)
```

This is the recommended method for integrating the dataset into different reconstruction pipelines, ensuring that each technique receives the same input images and corresponding ground-truth geometry.

### Testing the Dataset

Run the loader tests from the repository root:

```bash
pytest tests/test_ablation_dataset.py
```

The tests verify product lookup, image selection, mesh loading, and missing-file handling using sample data. If the downloaded dataset is available in `data/abo/`, an additional test checks the actual dataset files.