"""Local-only inference. Downloads belong to scripts/prepare_models.py."""
from functools import lru_cache
import os
from threading import RLock
from config import CLIP_MODEL, CACHE_DIR, TORCH_THREADS

MODEL_LOCK = RLock()

def configure_torch():
    import torch
    torch.set_num_threads(max(1, TORCH_THREADS))
    return torch

def device():
    torch = configure_torch()
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@lru_cache(maxsize=1)
def clip_components():
    # Transformers 5 otherwise launches a remote conversion thread even when
    # loading a cached .bin with local_files_only=True.
    os.environ['DISABLE_SAFETENSORS_CONVERSION'] = '1'
    from transformers import CLIPModel, CLIPProcessor
    try:
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL, cache_dir=str(CACHE_DIR), local_files_only=True)
        # Use the explicitly prepared checkpoint. This also disables automatic
        # background safetensors-conversion requests in newer Transformers.
        model = CLIPModel.from_pretrained(CLIP_MODEL, cache_dir=str(CACHE_DIR),
                                          local_files_only=True, use_safetensors=False)
    except (OSError, ValueError) as exc:
        raise RuntimeError('CLIP is not prepared. Run python scripts/prepare_models.py, then python scripts/build_index.py --mode all.') from exc
    model.to(device()).eval()
    return processor, model

def normalized_features(output):
    """Transformers 4 returns tensors; Transformers 5 returns pooled features."""
    torch = configure_torch()
    vector = output if torch.is_tensor(output) else output.pooler_output
    return torch.nn.functional.normalize(vector, dim=-1).detach().cpu().numpy()
