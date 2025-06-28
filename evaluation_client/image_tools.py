import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a float image in [0,1] to uint8 [0,255].

    Keeping everything in uint8 greatly reduces the payload size
    when transmitting frames.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


# --------------------------------------------------------------------------- #
#  OLD API (kept for backward-compatibility)                                  #
# --------------------------------------------------------------------------- #
def resize_with_pad(
    images: np.ndarray,
    height: int,
    width: int,
    method: int = Image.BILINEAR,
) -> np.ndarray:
    """Batch resize **with** zero-padding (unchanged from original)."""
    if images.shape[-3:-1] == (height, width):
        return images

    orig_shape = images.shape
    images = images.reshape(-1, *orig_shape[-3:])
    resized = np.stack(
        [_resize_with_pad_pil(Image.fromarray(im), height, width, method) for im in images]
    )
    return resized.reshape(*orig_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(
    image: Image.Image,
    height: int,
    width: int,
    method: int,
) -> Image.Image:
    """Helper for `resize_with_pad` (unchanged from original)."""
    cur_w, cur_h = image.size
    if (cur_w, cur_h) == (width, height):
        return image

    ratio = max(cur_w / width, cur_h / height)
    new_h, new_w = int(cur_h / ratio), int(cur_w / ratio)
    resized = image.resize((new_w, new_h), resample=method)

    canvas = Image.new(resized.mode, (width, height), 0)
    pad_h = (height - new_h) // 2
    pad_w = (width - new_w) // 2
    canvas.paste(resized, (pad_w, pad_h))
    return canvas


# --------------------------------------------------------------------------- #
#  NEW API – pure resize (⤵ no zero-padding, ignores aspect ratio)            #
# --------------------------------------------------------------------------- #
def resize(
    img: np.ndarray,
    height: int,
    width: int,
    method: int = Image.BILINEAR,
) -> np.ndarray:
    """Resize a **single** image to (height, width) with no padding.

    Aspect ratio is **not** preserved – the image is simply stretched /
    squashed to fit the target resolution.  This matches the new policy-
    server contract.
    """
    if img is None:
        return None
    if img.shape[0:2] == (height, width):
        return img
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((width, height), resample=method)
    return np.array(pil_img)
