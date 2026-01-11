"""
Lightweight preprocessing helpers to turn raw datasets into the HDF5 tensors
expected by `wavesfm.main_finetune_multi`.

Each function writes a single HDF5 file so others can fine-tune without
shipping raw data.
"""

from wavesfm.preprocessing.iq import preprocess_iq_task


def _missing(name: str):
    def _stub(*args, **kwargs):  # pragma: no cover - used only when deps are missing
        raise ImportError(f"{name} is unavailable because its dependencies are not installed.")
    return _stub


try:
    from preprocessing.preprocess_csi_sensing_cache import preprocess_csi_sensing as preprocess_sensing  # type: ignore
except Exception:  # pragma: no cover
    preprocess_sensing = _missing("preprocess_sensing")

try:
    from preprocessing.preprocess_radio_sig_cache import preprocess_radio_sig as preprocess_radio_signals  # type: ignore
except Exception:  # pragma: no cover
    preprocess_radio_signals = _missing("preprocess_radio_signals")

try:
    from preprocessing.preprocess_positioning_cache import preprocess_positioning  # type: ignore
except Exception:  # pragma: no cover
    preprocess_positioning = _missing("preprocess_positioning")

try:
    from preprocessing.preprocess_radcom_ota import preprocess_radcom_ota  # type: ignore
except Exception:  # pragma: no cover
    preprocess_radcom_ota = _missing("preprocess_radcom_ota")

try:
    from preprocessing.preprocess_uwb_loc import preprocess as preprocess_uwb  # type: ignore
except Exception:  # pragma: no cover
    preprocess_uwb = _missing("preprocess_uwb")

__all__ = [
    "preprocess_iq_task",
    "preprocess_sensing",
    "preprocess_radio_signals",
    "preprocess_positioning",
    "preprocess_radcom_ota",
    "preprocess_uwb",
]
