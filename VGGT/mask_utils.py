from pathlib import Path
import numpy as np
from PIL import Image

def load_mask(image_path, target_shape=None):
    p = Path(image_path)
    mask_path = p.parents[3] / "masks" / p.parent.parent.name / p.parent.name / (p.stem + ".png")
    mask = Image.open(mask_path).convert("L")
    w, h = mask.size
    new_h = round(h * (518 / w) / 14) * 14
    mask = mask.resize((518, new_h), Image.Resampling.NEAREST)
    if new_h > 518:
        top = (new_h - 518) // 2
        mask = mask.crop((0, top, 518, top + 518))
    result = np.asarray(mask) > 127

    if target_shape is not None:
        th, tw = map(int, target_shape)
        h, w = result.shape
        if h > th or w > tw:
            raise ValueError((result.shape, target_shape))
        dh, dw = th - h, tw - w
        result = np.pad(
            result,
            ((dh // 2, dh - dh // 2),
             (dw // 2, dw - dw // 2)),
            constant_values=False
        )

    return result
