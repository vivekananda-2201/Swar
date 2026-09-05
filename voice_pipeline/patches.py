"""
Automatic runtime patches for upstream libraries (nano_parakeet).

Ensures that voice_pipeline works out-of-the-box in any fresh virtual environment
without requiring manual modifications to site-packages.

Patches applied:
1. nano_parakeet.model.tdt_greedy_decode:
   Fixes upstream bug where return_timestamps=True on CPU only returned (tokens, token_frames)
   instead of (tokens, token_frames, enc_len), causing ValueError: not enough values to unpack.
2. nano_parakeet._loader.load_nemo_state_dict:
   Fixes upstream bug where 2.4GB was written into /tmp (tmpfs RAM) on every load, causing
   out-of-memory crashes. Instead, caches converted weights permanently in ~/.cache/nano_parakeet/
   on physical SSD for instant ~5s startup.
"""

import inspect
import logging
import os
import zipfile
import torch

logger = logging.getLogger(__name__)

_PATCHES_APPLIED = False


def _patch_loader():
    try:
        import nano_parakeet._loader as loader

        orig_loader = loader.load_nemo_state_dict

        def cached_load_nemo_state_dict(nemo_path: str, map_location='cpu') -> dict:
            cache_dir = os.path.expanduser("~/.cache/nano_parakeet")
            os.makedirs(cache_dir, exist_ok=True)

            nemo_stat = os.stat(nemo_path)
            cache_file = os.path.join(
                cache_dir, f"parakeet_tdt_0.6b_{int(nemo_stat.st_mtime)}_{nemo_stat.st_size}.pt"
            )

            # 1. Load directly from persistent SSD cache if present
            if os.path.isfile(cache_file) and os.path.getsize(cache_file) > 0:
                return torch.load(cache_file, map_location=map_location, weights_only=False)

            # 2. Extract and write to persistent cache on SSD (never overflows /tmp)
            z = zipfile.ZipFile(nemo_path)
            entries = [n for n in z.namelist() if n.startswith('model_weights/')]
            tmp_pt = cache_file + ".tmp"
            try:
                writer = torch._C.PyTorchFileWriter(tmp_pt)
                for entry in entries:
                    name = entry[len('model_weights/'):]
                    if not name:
                        continue
                    data = z.read(entry)
                    writer.write_record(name, data, len(data))
                writer.write_end_of_file()
                del writer
                os.replace(tmp_pt, cache_file)
                state_dict = torch.load(cache_file, map_location=map_location, weights_only=False)
            except Exception:
                if os.path.exists(tmp_pt):
                    os.remove(tmp_pt)
                raise
            return state_dict

        loader.load_nemo_state_dict = cached_load_nemo_state_dict
        logger.debug("nano_parakeet._loader cached state dict patch active.")
    except Exception as e:
        logger.warning(f"Could not apply nano_parakeet loader patch: {e}")


def _patch_model():
    try:
        import nano_parakeet.model as model_mod

        orig_decode = model_mod.tdt_greedy_decode

        def safe_tdt_greedy_decode(*args, **kwargs):
            res = orig_decode(*args, **kwargs)
            return_timestamps = kwargs.get("return_timestamps", False)
            if not return_timestamps and len(args) >= 6:
                return_timestamps = args[5]

            # If upstream only returned (tokens, token_frames), append enc_len
            if return_timestamps and isinstance(res, tuple) and len(res) == 2:
                # enc_len is the length of encoder output
                encoder_out = args[1] if len(args) > 1 else kwargs.get("encoder_out")
                enc_len = encoder_out.shape[1] if hasattr(encoder_out, "shape") else 0
                return res[0], res[1], enc_len
            return res

        model_mod.tdt_greedy_decode = safe_tdt_greedy_decode
        logger.debug("nano_parakeet.model tdt_greedy_decode patch active.")
    except Exception as e:
        logger.warning(f"Could not apply nano_parakeet model patch: {e}")


def apply_patches():
    """Applies all runtime compatibility patches in memory."""
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return
    _patch_loader()
    _patch_model()
    _PATCHES_APPLIED = True
