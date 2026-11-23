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
- **Chamfer L1** is the mean of accuracy and completeness.

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
