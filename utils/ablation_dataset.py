"""ablation_dataset.py
Reusable access to ABO image ablations and corresponding ground-truth meshes.

@author: Devin
"""

# ==============================================================================
# Imports
# ==============================================================================

import csv
from pathlib import Path

from PIL import Image
import trimesh


# ==============================================================================
# Amazon Berkeley Objects Dataset
# ==============================================================================


class ABODataset:
    """Access selected ABO products, evenly spaced image views, and GT meshes.

    The default data location is ``data/abo/`` relative to the repository root.
    Images are stored in azimuth order as ``00.jpg`` through ``71.jpg``.

    Attributes:
        TOTAL_VIEWS (int): Number of available views per selected product.
        data_dir (Path): Root directory containing the selected ABO dataset.
        products (list[dict[str, str]]): Product records from the selection CSV.

    Example:
        >>> dataset = ABODataset()
        >>> product = dataset.list_products(category="CHAIR")[0]
        >>> sample = dataset.get_product(product["spin_id"], num_images=18)
        >>> len(sample["image_paths"])
        18
    """

    TOTAL_VIEWS = 72

    # ==============================================================================
    # Initialization and Product Lookup
    # ==============================================================================

    def __init__(self, data_dir=None):
        """Read the selected-product manifest from the dataset directory.

        Args:
            data_dir (str | Path | None): ABO dataset location. If omitted,
                uses ``<repo_root>/data/abo``.

        Raises:
            FileNotFoundError: If ``selected_products.csv`` does not exist.
        """
        if data_dir is None:
            # Resolve from this file's location, not the current working directory.
            data_dir = Path(__file__).resolve().parents[1] / "data" / "abo"

        self.data_dir = Path(data_dir).resolve()
        csv_path = self.data_dir / "selected_products.csv"

        with csv_path.open(newline="", encoding="utf-8") as file:
            self.products = list(csv.DictReader(file))

        # Support lookup by either the image sequence ID or Amazon item ID.
        self._by_spin = {row["spin_id"]: row for row in self.products}
        self._by_item = {row["item_id"]: row for row in self.products}

    # Return copies so calling code cannot accidentally change dataset records.
    def list_products(self, category=None):
        """List selected products, optionally filtering by category.

        Args:
            category (str | None): ABO product type, such as ``"CHAIR"``.

        Returns:
            list[dict[str, str]]: Matching product metadata records.
        """
        return [
            row.copy()
            for row in self.products
            if category is None or row["product_type"] == category
        ]

    # Resolve both identifiers through the same product metadata record.
    def _product(self, product_id):
        """Look up a selected product by spin ID or item ID.

        Args:
            product_id (str): Product ``spin_id`` or ``item_id``.

        Returns:
            dict[str, str]: Matched metadata record.

        Raises:
            KeyError: If the ID is absent from the selected-products CSV.
        """
        row = self._by_spin.get(str(product_id)) or self._by_item.get(
            str(product_id)
        )
        if row is None:
            raise KeyError(
                f"Product not found in selected_products.csv: {product_id}"
            )
        return row

    # ==============================================================================
    # Image Selection and Loading
    # ==============================================================================

    # Spread indices across the complete rotation without repeating viewpoints.
    @classmethod
    def image_indices(cls, num_images=72):
        """Select approximately equidistant indices from the 72 image views.

        Counts of 72, 36, 18, and 9 use strides of 1, 2, 4, and 8. For
        other counts, the gaps between selected indices differ by at most one.

        Args:
            num_images (int): Number of viewpoints to select, from 1 to 72.

        Returns:
            list[int]: Ascending, zero-based rotation indices.

        Raises:
            ValueError: If the requested count is not an integer in [1, 72].
        """
        if (not isinstance(num_images, int) or
                not 1 <= num_images <= cls.TOTAL_VIEWS):
            raise ValueError("num_images must be an integer between 1 and 72")

        return [
            (index * cls.TOTAL_VIEWS) // num_images
            for index in range(num_images)
        ]

    # Map selected indices to the shared on-disk image directory.
    def get_image_paths(self, product_id, num_images=72):
        """Return existing image paths in rotation order without loading them.

        Args:
            product_id (str): Selected product spin ID or item ID.
            num_images (int): Number of evenly spaced views to use.

        Returns:
            list[Path]: Paths to the selected JPEGs.

        Raises:
            KeyError: If the product ID is unknown.
            ValueError: If the image count is invalid.
            FileNotFoundError: If any requested image is absent.
        """
        row = self._product(product_id)
        image_dir = (
            self.data_dir / "images" / row["product_type"] / row["spin_id"]
        )
        paths = [
            image_dir / f"{index:02d}.jpg"
            for index in self.image_indices(num_images)
        ]

        # Fail early rather than letting downstream reconstruction fail later.
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} image(s), first: {missing[0]}"
            )

        return paths

    # Load RGB pixels only when a pipeline needs in-memory image objects.
    def load_images(self, product_id, num_images=72):
        """Load selected image views as independent PIL RGB images.

        Args:
            product_id (str): Selected product spin ID or item ID.
            num_images (int): Number of evenly spaced views to use.

        Returns:
            list[Image.Image]: PIL images ordered by rotation index.

        Raises:
            FileNotFoundError: If a requested image does not exist.
            OSError: If an image cannot be decoded.
        """
        images = []
        for path in self.get_image_paths(product_id, num_images):
            with Image.open(path) as image:
                images.append(image.convert("RGB"))

        return images

    # ==============================================================================
    # Ground-Truth Mesh Access
    # ==============================================================================

    # Find the corresponding ground-truth asset via the product's 3dmodel_id.
    def get_mesh_path(self, product_id, mesh_format="ply"):
        """Find the selected product's ground-truth PLY or original GLB.

        Args:
            product_id (str): Selected product spin ID or item ID.
            mesh_format (str): ``"ply"``, ``"glb"``, or ``"auto"``. Auto
                prefers a converted PLY and falls back to the original GLB.

        Returns:
            Path: Existing ground-truth mesh file.

        Raises:
            KeyError: If the product ID is unknown.
            ValueError: If no model ID exists or mesh_format is invalid.
            FileNotFoundError: If no requested mesh file is present.
        """
        row = self._product(product_id)
        model_id = row.get("3dmodel_id", "")
        if not model_id:
            raise ValueError(f"No 3dmodel_id for {product_id}")

        if mesh_format not in ("ply", "glb", "auto"):
            raise ValueError("mesh_format must be 'ply', 'glb', or 'auto'")

        candidates = {
            "ply": self.data_dir / "meshes" / f"{model_id}.ply",
            "glb": self.data_dir / "models" / f"{model_id}.glb",
        }
        order = ("ply", "glb") if mesh_format == "auto" else (mesh_format,)

        for candidate_format in order:
            if candidates[candidate_format].is_file():
                return candidates[candidate_format]

        raise FileNotFoundError(
            f"No {mesh_format} mesh for {product_id}: {model_id}"
        )

    # Apply scene node transforms when loading a GLB with multiple components.
    def load_mesh(self, product_id, mesh_format="ply"):
        """Load the product geometry as a combined Trimesh object.

        Args:
            product_id (str): Selected product spin ID or item ID.
            mesh_format (str): ``"ply"``, ``"glb"``, or ``"auto"``.

        Returns:
            trimesh.Trimesh: Combined geometry with scene transforms applied.

        Raises:
            FileNotFoundError: If the requested mesh does not exist.
        """
        mesh_path = self.get_mesh_path(product_id, mesh_format)
        return trimesh.load_scene(mesh_path).to_mesh()

    # ==============================================================================
    # Combined Product Access
    # ==============================================================================

    # Provide a common input record for every reconstruction implementation.
    def get_product(self, product_id, num_images=72, mesh_format="ply"):
        """Return image paths, view indices, mesh path, and product metadata.

        Args:
            product_id (str): Selected product spin ID or item ID.
            num_images (int): Number of evenly spaced image views to use.
            mesh_format (str): Requested mesh file type.

        Returns:
            dict: Product IDs, category, selected indices and image paths,
                and the ground-truth mesh path.

        Raises:
            KeyError: If the product ID is unknown.
            ValueError: If the image count or mesh format is invalid.
            FileNotFoundError: If requested image or mesh files are missing.
        """
        row = self._product(product_id)
        return {
            "spin_id": row["spin_id"],
            "item_id": row["item_id"],
            "product_type": row["product_type"],
            "image_indices": self.image_indices(num_images),
            "image_paths": self.get_image_paths(product_id, num_images),
            "mesh_path": self.get_mesh_path(product_id, mesh_format),
        }