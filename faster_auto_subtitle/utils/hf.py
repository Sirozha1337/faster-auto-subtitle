import os
import logging
from contextlib import contextmanager
from typing import Callable, Iterator, TypeVar
from huggingface_hub import constants
from ..errors import ModelNotCachedError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# For models that only have pytorch_model.bin (most OPUS-MT ones), transformers otherwise starts a
# background thread that asks the Hub for a safetensors conversion PR and downloads that copy too.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

# Kept inside the hub cache so it persists wherever the models do (e.g. the Docker volume).
CACHE_DIR = os.path.join(constants.HF_HUB_CACHE, "faster_auto_subtitle")


def set_offline() -> None:
    """Forbid all Hugging Face Hub requests for the rest of the process."""
    constants.HF_HUB_OFFLINE = True


def is_offline() -> bool:
    """True if --offline was passed or HF_HUB_OFFLINE is set in the environment."""
    return constants.HF_HUB_OFFLINE


@contextmanager
def _offline_mode() -> Iterator[None]:
    previous = constants.HF_HUB_OFFLINE
    constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        constants.HF_HUB_OFFLINE = previous


def load_cached_first(load: Callable[[], T], description: str) -> T:
    """
    Calls `load` with Hub requests disabled, so a cached model is used without contacting
    Hugging Face. Only if that fails, and we're not offline, `load` is retried online.
    """
    try:
        with _offline_mode():
            return load()
    except OSError as exc:
        # LocalEntryNotFoundError and transformers' "couldn't find in cache" are both OSErrors.
        if is_offline():
            raise ModelNotCachedError(
                f"{description} is not in the local cache and offline mode is on (--offline or HF_HUB_OFFLINE). "
                "Run once without --offline to download it.") from exc

    logger.info("%s is not cached, downloading it from Hugging Face.", description)
    return load()
