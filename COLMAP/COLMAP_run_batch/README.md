# COLMAP batch benchmark — PyCharm / Windows
Prepared against splatify_project commit 959bdae976a75ff2a32c98cf3c9220dd916fd011.
Place this COLMAP folder at the repository root alongside utils/, data/, 3DGS/, VGGT/.
Python 3.10+ recommended. Install: python -m pip install pillow trimesh numpy
COLMAP is a separate program; pip dependencies do not install COLMAP or CUDA.

## Setup
1. Clone/download the repository and open its root in PyCharm. Create your own Git branch.
2. Download the NEW 20-product archive from the team's shared Google Drive.
   Extract so data/abo/selected_products.csv exists. Avoid data/abo/abo nesting.
   All 72 images per object and the reference meshes must be present.
   The repository docs still describe the OLD 100-product dataset; do not use
   the creation notebook blindly or choose the first 20 products yourself.
3. Edit COLMAP/config.json. Use forward slashes in Windows paths.
   colmap_executable must point to the official COLMAP.bat launcher (or executable).
   data_dir and output_dir are resolved relative to repository root.
4. Right-click COLMAP/run_batch.py -> Run. Default mode=check validates the dataset,
   reference meshes and command availability, but does NOT prove CUDA works.
5. Set mode=pilot; set pilot_product_id to a selected_products.csv item_id or spin_id
   (empty chooses the first row). Run again: three independent experiments, 18/36/72.
6. Inspect logs, registration counts and fused.ply/mesh in a viewer.
   If the environment is OK, set mode=all and run. Already completed pilot runs skip.
   Keep parameters, COLMAP binary, dataset and output directory unchanged.

## Protocol
Uses the repository ABODataset for nested uniformly spaced view subsets.
SIFT/exhaustive GPU matching, SIMPLE_RADIAL, independent intrinsics by default
(single_camera=0), incremental mapper, largest component by registered views
then point count, undistortion at 2000 px, geometric PatchMatch, geometric fusion,
optional Poisson depth=13, point_weight=1, trim=5.
Other parameters follow the installed COLMAP defaults, captured in protocol.json.
This is a fixed batch protocol, not a claim of reproducing all historical GUI settings.
If changing parameters or software, choose a NEW output directory.
Minimum three registered images and nonzero sparse points are required to run dense
steps; this is our explicit operational rule, not COLMAP's universal definition of success.
Full registration or nonempty mesh does NOT imply geometric accuracy.

## Output
data/colmap_runs_v1/summary.csv (latest attempt per configuration)
data/colmap_runs_v1/protocol.json (config, manifest hash, binary help/defaults)
data/colmap_runs_v1/<spin_id>/<18|36|72>/attempt_<timestamp>/
  selected_images.json, database.db, sparse/, images/, dense/
  features.log, matching.log, sfm.log, undistortion.log, stereo.log,
  fusion.log, meshing.log, result.json
Stage times measure wall time for subprocesses including launch overhead.
Total time includes input copying and stage overhead. Failed stage time is retained.
Attempts restart fresh; completed configurations skip. Old failed attempts stay on disk.
Geometry metrics remain null: integrate Devin's evaluation once provided.
Do not compare full-resolution timing across different machines without labeling hardware.

## Stop / recovery
Only run ONE batch per GPU/output directory. Ctrl+C in a terminal stops the current
child process tree and saves interrupted status. PyCharm's force Stop or power loss
may prevent cleanup; first verify no COLMAP process remains before restarting.
Remove RUNNING.lock ONLY after verifying the previous batch has stopped.
A subsequent run starts an incomplete/failed configuration from scratch, preserving
its old attempt folder. This trades additional computation for clean timing.
Do not edit/delete images during a run. Do not keep summary.csv open in Excel while
the batch writes it. Each stage has a configurable 7200-second timeout.
If CUDA/DLL/CLI errors repeat, stop and fix the environment instead of running 60 failures.
An empty initial reconstruction is an algorithm failure; missing CUDA is an environment
failure: retain the error log and distinguish these in the final report.

## Git / sharing
Commit only scripts, portable config, README and selected result tables.
Large outputs stay in ignored data/ or the shared Drive; never commit PLY/DB/archive files.
Do not commit a config containing your local absolute paths; use the example placeholder.
No uploading or cloud job submission is performed by this script.

## Validation scope
Python syntax and a synthetic mocked-COLMAP end-to-end pilot were checked.
Actual Windows launcher/CUDA reconstruction must be validated on your machine.
