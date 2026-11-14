
import gc
import glob
import os
import time

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def main():

    # ==================================================
    # 1. Check GPU and CUDA
    # ==================================================

    print("PyTorch:", torch.__version__, flush=True)
    print("CUDA available:", torch.cuda.is_available(), flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = "cuda"

    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    total_memory = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
    )

    print("Total GPU memory:", round(total_memory, 2), "GiB", flush=True)

    # ==================================================
    # 2. Select two test images
    # ==================================================

    image_paths = sorted(
        p for p in glob.glob("examples/kitchen/images/*")
        if p.lower().endswith((".png", ".jpg", ".jpeg"))
    )[:2]

    if len(image_paths) < 2:
        raise RuntimeError("Not enough test images!")

    print("Input images:", image_paths, flush=True)

    # ==================================================
    # 3. Locate downloaded VGGT weights
    # ==================================================

    print("Locating cached VGGT weights...", flush=True)

    path = hf_hub_download(
        repo_id="facebook/VGGT-1B",
        filename="model.safetensors",
        local_files_only=True
    )

    # ==================================================
    # 4. Create VGGT model on CPU
    # ==================================================

    print("Creating VGGT model...", flush=True)

    model = VGGT()

    # ==================================================
    # 5. Load weights using pread
    # ==================================================

    print("Loading weights with pread...", flush=True)

    missing, unexpected = load_model(
        model,
        path,
        device="cpu",
        backend="pread"
    )

    print("Missing:", missing, flush=True)
    print("Unexpected:", unexpected, flush=True)

    if missing or unexpected:
        raise RuntimeError("Model weights do not match!")

    model.eval()

    print("CPU model loaded successfully!", flush=True)

    # ==================================================
    # 6. Move FP32 model to GPU
    # ==================================================

    print("Moving model to GPU...", flush=True)

    torch.cuda.empty_cache()

    # Keep model weights in FP32
    # Do NOT use model.half()
    model = model.to(device)

    print("Model on GPU!", flush=True)

    # ==================================================
    # 7. Preprocess input images
    # ==================================================

    print("Preprocessing images...", flush=True)

    images = load_and_preprocess_images(image_paths)

    # Keep image inputs in FP32
    images = images.to(device)

    print("Image shape:", images.shape, flush=True)

    # ==================================================
    # 8. Select mixed precision
    # ==================================================

    gpu_capability = torch.cuda.get_device_capability(0)

    if gpu_capability[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    print("Autocast dtype:", dtype, flush=True)

    # ==================================================
    # 9. Run VGGT inference
    # ==================================================

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    print("Starting inference...", flush=True)

    start_time = time.perf_counter()

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=dtype
        ):
            predictions = model(images)

    torch.cuda.synchronize()

    inference_time = time.perf_counter() - start_time

    print("Inference successful!", flush=True)

    print(
        "Inference time:",
        round(inference_time, 2),
        "seconds",
        flush=True
    )

    print(
        "Prediction keys:",
        list(predictions.keys()),
        flush=True
    )

    # ==================================================
    # 10. Record GPU memory usage
    # ==================================================

    peak_memory = (
        torch.cuda.max_memory_allocated() / 1024**3
    )

    print(
        "Peak GPU memory:",
        round(peak_memory, 2),
        "GiB",
        flush=True
    )

    # ==================================================
    # 11. Save predictions to CPU
    # ==================================================

    print("Saving predictions...", flush=True)

    results = {}

    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            results[key] = value.detach().float().cpu()

    os.makedirs("outputs", exist_ok=True)

    output_path = "outputs/first_test.pt"

    torch.save(results, output_path)

    print(
        "Results saved to:",
        output_path,
        flush=True
    )

    # ==================================================
    # 12. Clean up
    # ==================================================

    del predictions
    del images
    del results
    del model

    gc.collect()
    torch.cuda.empty_cache()

    print("VGGT test completed!", flush=True)


if __name__ == "__main__":
    main()
