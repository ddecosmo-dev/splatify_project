"""test_ablation_dataset.py
Test the shared ABO dataset loader with synthetic and downloaded assets.

Copyright (c) 2026 Devin DeCosmo.
@author: Devin
"""

# ==============================================================================
# Imports
# ==============================================================================

import csv
from pathlib import Path

import pytest
import trimesh
from PIL import Image

from utils.ablation_dataset import ABODataset


# ==============================================================================
# Test Dataset Setup
# ==============================================================================


@pytest.fixture
def dataset(tmp_path):
    """Create a small, complete ABO dataset without downloading any assets.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.

    Returns:
        ABODataset: Loader pointing to two synthetic products.
    """
    data_dir = tmp_path / "data" / "abo"
    products = [
        {
            "item_id": "item_01",
            "spin_id": "spin_01",
            "3dmodel_id": "mesh_01",
            "product_type": "CHAIR",
        },
        {
            "item_id": "item_02",
            "spin_id": "spin_02",
            "3dmodel_id": "mesh_02",
            "product_type": "LAMP",
        },
    ]

    data_dir.mkdir(parents=True)
    with (data_dir / "selected_products.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=products[0].keys())
        writer.writeheader()
        writer.writerows(products)

    # Match the real folder convention: images/<category>/<spin_id>/00.jpg.
    for product in products:
        image_dir = (
            data_dir / "images" / product["product_type"] / product["spin_id"]
        )
        image_dir.mkdir(parents=True)
        for index in range(72):
            image = Image.new("RGB", (4, 4), (index, 0, 0))
            image.save(image_dir / f"{index:02d}.jpg")

        # The converted PLY represents the ground-truth geometry.
        mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
        mesh_dir = data_dir / "meshes"
        mesh_dir.mkdir(exist_ok=True)
        mesh.export(mesh_dir / f"{product['3dmodel_id']}.ply")

    # The GLB scene has a transform, which load_mesh() should apply.
    scene = trimesh.Scene()
    transform = trimesh.transformations.translation_matrix((10, 0, 0))
    scene.add_geometry(trimesh.creation.box(), transform=transform)
    model_dir = data_dir / "models"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "mesh_01.glb").write_bytes(scene.export(file_type="glb"))

    return ABODataset(data_dir=data_dir)


# ==============================================================================
# Image Selection Tests
# ==============================================================================


@pytest.mark.parametrize(
    ("count", "stride"),
    [(72, 1), (36, 2), (18, 4), (9, 8)],
)
def test_standard_image_indices(count, stride):
    """Verify nested, uniformly spaced image selection for each ablation."""
    assert ABODataset.image_indices(count) == list(range(0, 72, stride))


def test_other_image_counts_are_evenly_spaced():
    """Verify approximately equal circular spacing for nonstandard counts."""
    indices = ABODataset.image_indices(10)
    circular_gaps = [
        (indices[(index + 1) % len(indices)] - indices[index]) % 72
        for index in range(len(indices))
    ]

    assert len(indices) == len(set(indices)) == 10
    assert max(circular_gaps) - min(circular_gaps) <= 1


@pytest.mark.parametrize("count", [0, -1, 73, 1.5, "18"])
def test_invalid_image_count(count):
    """Reject unsupported image counts before accessing the filesystem."""
    with pytest.raises(ValueError):
        ABODataset.image_indices(count)


def test_get_image_paths(dataset):
    """Return the requested ordered paths, using either product identifier."""
    paths = dataset.get_image_paths("spin_01", num_images=9)

    assert len(paths) == 9
    assert [path.stem for path in paths] == [
        f"{index:02d}" for index in range(0, 72, 8)
    ]
    assert paths == dataset.get_image_paths("item_01", num_images=9)
    assert all(path.is_file() for path in paths)


def test_load_images(dataset):
    """Decode selected image files as in-memory RGB PIL images."""
    images = dataset.load_images("spin_01", num_images=9)

    assert len(images) == 9
    assert all(image.mode == "RGB" for image in images)
    assert all(image.size == (4, 4) for image in images)


# ==============================================================================
# Product and Mesh Tests
# ==============================================================================


def test_list_products(dataset):
    """Return all products or filter by category without exposing internals."""
    assert len(dataset.list_products()) == 2
    assert len(dataset.list_products(category="CHAIR")) == 1
    assert dataset.list_products(category="RUG") == []

    result = dataset.list_products(category="CHAIR")
    result[0]["product_type"] = "MODIFIED"
    assert dataset.list_products(category="CHAIR")[0]["product_type"] == "CHAIR"


def test_mesh_lookup_and_loading(dataset):
    """Locate converted meshes and original GLBs, applying scene transforms."""
    assert dataset.get_mesh_path("spin_01").name == "mesh_01.ply"
    assert dataset.get_mesh_path("item_01", "glb").name == "mesh_01.glb"
    assert dataset.get_mesh_path("spin_01", "auto").suffix == ".ply"

    ply_mesh = dataset.load_mesh("spin_01")
    glb_mesh = dataset.load_mesh("spin_01", mesh_format="glb")

    assert isinstance(ply_mesh, trimesh.Trimesh)
    assert len(ply_mesh.faces) > 0
    assert isinstance(glb_mesh, trimesh.Trimesh)
    assert glb_mesh.bounding_box.centroid[0] == pytest.approx(10.0)


def test_get_product(dataset):
    """Return consistent image indices, paths, metadata, and mesh path."""
    sample = dataset.get_product("spin_01", num_images=18)

    assert sample["spin_id"] == "spin_01"
    assert sample["item_id"] == "item_01"
    assert sample["product_type"] == "CHAIR"
    assert sample["image_indices"] == list(range(0, 72, 4))
    assert len(sample["image_paths"]) == 18
    assert sample["mesh_path"].name == "mesh_01.ply"


# ==============================================================================
# Error Handling Tests
# ==============================================================================


def test_missing_assets_and_unknown_products(dataset):
    """Fail clearly for missing images, meshes, and invalid product IDs."""
    with pytest.raises(KeyError):
        dataset.get_product("unknown", num_images=9)

    image = dataset.get_image_paths("spin_01", num_images=72)[0]
    image.unlink()
    with pytest.raises(FileNotFoundError):
        dataset.get_image_paths("spin_01", num_images=9)

    with pytest.raises(ValueError):
        dataset.get_mesh_path("spin_01", mesh_format="obj")

    mesh = dataset.get_mesh_path("spin_02")
    mesh.unlink()
    with pytest.raises(FileNotFoundError):
        dataset.get_mesh_path("spin_02")


def test_missing_manifest(tmp_path):
    """Reject a dataset location without selected_products.csv."""
    with pytest.raises(FileNotFoundError):
        ABODataset(data_dir=tmp_path)


# ==============================================================================
# Real Dataset Smoke Test (Optional)
# ==============================================================================


def test_downloaded_dataset_if_available():
    """Check downloaded images and GT assets if data/abo exists in the repo.

    Skips automatically for CI runs or teammates without downloaded ABO data.
    """
    data_dir = Path(__file__).resolve().parents[1] / "data" / "abo"
    if not (data_dir / "selected_products.csv").is_file():
        pytest.skip("No downloaded ABO dataset found in data/abo")

    real_dataset = ABODataset(data_dir=data_dir)
    products = real_dataset.list_products()
    assert products, "selected_products.csv is empty"

    # Validate the full image set and matching mesh for each selected product.
    for product in products:
        spin_id = product["spin_id"]
        assert len(real_dataset.get_image_paths(spin_id, 72)) == 72
        assert len(real_dataset.get_image_paths(spin_id, 18)) == 18
        assert real_dataset.get_mesh_path(spin_id).is_file()

    # Decode one image and mesh to catch invalid files, not only absent paths.
    first_id = products[0]["spin_id"]
    assert len(real_dataset.load_images(first_id, 9)) == 9
    assert len(real_dataset.load_mesh(first_id).faces) > 0
