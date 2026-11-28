# Output Metrics

The shared evaluation code measures reconstruction outputs only. Runtime, GPU
memory, and other method-specific profiling remain with each reconstruction
implementation.

Each method should be evaluated with 72, 36, 18, and 9 evenly spaced images.

## Adapters

`utils.evaluation.adapters` converts each output into a deterministic point
sample without changing its native representation:

- triangle meshes are sampled uniformly on their surfaces;
- point clouds retain their native points and are deterministically
  subsampled when necessary; and
- 3DGS PLY files retain Gaussian centers after applying the configured
  activated-opacity threshold.

Point clouds and Gaussian splats are not converted into meshes. Meshing would
introduce a second reconstruction method into the comparison.

## Metrics

`utils.evaluation.metrics` first estimates the uniform scale correction needed
to match the reference. It records that correction and a symmetric percentage
error before applying it. The scaled prediction is then rigidly aligned to the
reference using PCA proposals and trimmed ICP.

The aligned output is evaluated in both directions:

- **accuracy** is the mean prediction-to-reference distance and penalizes
  floating or incorrect geometry;
- **completeness** is the mean reference-to-prediction distance and penalizes
  missing geometry; and
- **Chamfer L1** is the mean of accuracy and completeness; and
- **noise error** is the percentage of aligned prediction points farther than
  a chosen tolerance from the nearest sampled reference point. The default
  tolerance is 5% of the reference sample's bounding-box diagonal.

Distances are reported in reference units and normalized by the reference
bounding-box diagonal for comparisons across object sizes.

```python
from utils.evaluation import adapt_geometry, evaluate_geometry

prediction = adapt_geometry("outputs/3dgs/point_cloud.ply")
reference = adapt_geometry(
    "data/abo-selected-20/meshes/B01DN7NFRG.ply",
    kind="mesh",
)

result = evaluate_geometry(prediction.points, reference.points)
```

Noise error measures off-surface geometry, including both floaters and
systematic shape deviations. Read it alongside completeness to see whether a
lower outlier rate comes with missing geometry.

## VGGT batch evaluation

Extract the VGGT archive under `data/` and pass its top-level folder to the
runner. The folder's `manifest.tsv` lists every expected product and view
count. The runner resolves reference meshes through the ABO dataset manifest
and evaluates both raw and masked point clouds.

```bash
unzip -n data/VGGT_all20-20261128T165408Z-1-001.zip -d data
python -m app.evaluate_vggt data/VGGT_all20 \
  --dataset-dir data/abo-selected-20 \
  --output-dir outputs/evaluation/vggt-all20
```

- `summary.csv` has means and medians by cloud variant and view count. The
  `all` rows weight each product equally after averaging its available views.
- `per_object.csv` averages view counts for each product and ranks by shape
  error within each variant.
- `per_run.csv` has each individual cloud, ranked by shape error within its
  variant and view count. Rank 1 is the lowest shape error.

Shape error is normalized Chamfer L1 expressed as a percentage. Use
`--noise-threshold` to set the tolerance as a fraction of the reference
diagonal. Each cloud is aligned independently using the shared evaluator.
