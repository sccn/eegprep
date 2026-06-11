"""Shared PAC limitation text for core EEGPrep helpers."""

PAC_UNSUPPORTED_MESSAGE = (
    "EEGPrep does not implement standalone phase-amplitude coupling in core. "
    "EEGLAB pac/pac_cont and STUDY PAC use PAC-specific analysis choices and "
    ".icapac/.datpac sidecar caches; add a tested EEGPrep-owned PAC backend "
    "before calling this helper."
)


def raise_pac_not_implemented() -> None:
    """Raise EEGPrep's explicit PAC limitation."""
    raise NotImplementedError(PAC_UNSUPPORTED_MESSAGE)


__all__ = ["PAC_UNSUPPORTED_MESSAGE", "raise_pac_not_implemented"]
