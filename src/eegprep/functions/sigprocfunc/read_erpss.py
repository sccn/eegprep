"""Read ERPSS ``.RAW`` and ``.RDF`` recordings."""

from __future__ import annotations

from pathlib import Path
import struct
from typing import BinaryIO, Any

import numpy as np


_FILE_HEADER_BYTES = 4096
_BLOCK_HEADER_BYTES = 512
_MAX_EVENTS_PER_BLOCK = 110
_CHANNEL_LABEL_OFFSET = 580
_CHANNEL_LABEL_BYTES = 8
_LITTLE_FILE_TAG = b"\x55\xaa\xb0\x00"
_LITTLE_BLOCK_TAG = 0x00F0AA55
_BIG_FILE_TAG = b"\xaa\x55\x00\xb0"
_BIG_BLOCK_TAG = 0xAA5500F0


def read_erpss(filename: str | Path) -> tuple[np.ndarray, list[dict[str, int]], dict[str, Any]]:
    """Read an ERPSS recording into channel-major microvolt data.

    ERPSS files contain a 4096-byte recording header followed by 512-byte
    block headers and either interleaved signed 16-bit samples or ERPSS
    delta-compressed samples. Event latencies returned in ``sample_offset``
    are 1-based, matching EEG dataset event latencies.

    Args:
        filename: ERPSS ``.RAW`` or ``.RDF`` file.

    Returns:
        A ``(data, events, header)`` tuple. ``data`` has shape
        ``(nchans, nframes)`` and is scaled to microvolts when calibration is
        present. Each event contains ``sample_offset``, ``event_code``, and
        ``cond_code``.

    Raises:
        ValueError: If the file header, a block, or compressed data is invalid
            or truncated.
    """
    path = Path(filename)
    with path.open("rb") as stream:
        file_header = stream.read(_FILE_HEADER_BYTES)
        if len(file_header) != _FILE_HEADER_BYTES:
            raise ValueError("ERPSS file is shorter than its 4096-byte header")
        byteorder, prefix, block_tag = _byte_order(file_header[:4])
        compressed = bool(struct.unpack_from(f"{prefix}H", file_header, 4)[0])
        nchans = int(struct.unpack_from(f"{prefix}H", file_header, 6)[0])
        if nchans <= 0:
            raise ValueError("ERPSS header declares no channels")
        labels = _channel_labels(file_header, nchans)
        sample_interval = int(struct.unpack_from(f"{prefix}H", file_header, 552)[0])
        blocks, events, metadata = _read_blocks(
            stream,
            prefix=prefix,
            byteorder=byteorder,
            block_tag=block_tag,
            nchans=nchans,
            compressed=compressed,
        )

    if not blocks:
        raise ValueError("ERPSS file contains no data blocks")
    data = np.concatenate(blocks, axis=1).astype(float, copy=False)
    scale = _microvolt_scale(metadata)
    if scale is not None:
        data *= scale
    srate = _sampling_rate(metadata, sample_interval)
    header: dict[str, Any] = {
        "nchans": nchans,
        "nframes": int(data.shape[1]),
        "nblocks": len(blocks),
        "compressed": compressed,
        "byteorder": byteorder,
        "chanlabels": labels,
        "srate": srate,
        **metadata,
    }
    if scale is not None:
        header["rescaleuv"] = scale
    return data, events, header


def _byte_order(tag: bytes) -> tuple[str, str, int]:
    if tag == _LITTLE_FILE_TAG:
        return "little", "<", _LITTLE_BLOCK_TAG
    if tag == _BIG_FILE_TAG:
        return "big", ">", _BIG_BLOCK_TAG
    raise ValueError("File does not start with a recognized ERPSS byte-order tag")


def _channel_labels(header: bytes, nchans: int) -> list[str]:
    end = _CHANNEL_LABEL_OFFSET + nchans * _CHANNEL_LABEL_BYTES
    if end > len(header):
        return [f"Ch{index}" for index in range(1, nchans + 1)]
    labels = []
    for index in range(nchans):
        start = _CHANNEL_LABEL_OFFSET + index * _CHANNEL_LABEL_BYTES
        raw = header[start : start + _CHANNEL_LABEL_BYTES]
        label = raw.split(b"\x00", 1)[0].decode("latin-1").strip()
        labels.append(label or f"Ch{index + 1}")
    return labels


def _read_blocks(
    stream: BinaryIO,
    *,
    prefix: str,
    byteorder: str,
    block_tag: int,
    nchans: int,
    compressed: bool,
) -> tuple[list[np.ndarray], list[dict[str, int]], dict[str, int]]:
    blocks: list[np.ndarray] = []
    events: list[dict[str, int]] = []
    metadata: dict[str, int] = {}
    samples_before_block = 0
    while True:
        raw_header = stream.read(_BLOCK_HEADER_BYTES)
        if not raw_header:
            break
        if len(raw_header) != _BLOCK_HEADER_BYTES:
            raise ValueError(f"ERPSS block {len(blocks) + 1} has a truncated header")
        fields = struct.unpack_from(f"{prefix}I10H", raw_header)
        (
            tag,
            _record_type,
            block_nchans,
            _block_version,
            block_size,
            ndupsamp,
            nrun,
            err_detect,
            nlost,
            nevents,
            compressed_words,
        ) = fields
        block_number = len(blocks) + 1
        if tag != block_tag:
            raise ValueError(f"ERPSS block {block_number} has an invalid tag")
        if block_nchans != nchans:
            raise ValueError(f"ERPSS block {block_number} declares {block_nchans} channels; expected {nchans}")
        if block_size <= 0:
            raise ValueError(f"ERPSS block {block_number} has no samples")
        if nevents > _MAX_EVENTS_PER_BLOCK:
            raise ValueError(f"ERPSS block {block_number} declares more than 110 events")
        if not blocks:
            metadata = _block_metadata(raw_header, prefix)
        metadata.update(
            {
                "ndupsamp": int(ndupsamp),
                "nrun": int(nrun),
                "err_detect": int(err_detect),
                "nlost": int(nlost),
            }
        )
        events.extend(
            _block_events(
                raw_header,
                prefix=prefix,
                nevents=nevents,
                block_size=block_size,
                samples_before_block=samples_before_block,
                block_number=block_number,
            )
        )
        payload_bytes = int(compressed_words) * 2 if compressed else nchans * block_size * 2
        payload = stream.read(payload_bytes)
        if len(payload) != payload_bytes:
            raise ValueError(f"ERPSS block {block_number} has truncated sample data")
        if compressed:
            flat = _decompress(payload, nchans * block_size, nchans, byteorder)
        else:
            flat = np.frombuffer(payload, dtype=f"{prefix}i2", count=nchans * block_size).astype(np.int16, copy=False)
        blocks.append(flat.reshape(block_size, nchans).T)
        samples_before_block += block_size
    return blocks, events, metadata


def _block_metadata(header: bytes, prefix: str) -> dict[str, int]:
    amplif, clock_freq, divider, ad_range_mv, ad_bits, nsteps = struct.unpack_from(f"{prefix}6I", header, 24)
    return {
        "amplif": int(amplif),
        "clock_freq": int(clock_freq),
        "divider": int(divider),
        "ad_range_mv": int(ad_range_mv),
        "ad_bits": int(ad_bits),
        "nsteps": int(nsteps),
    }


def _block_events(
    header: bytes,
    *,
    prefix: str,
    nevents: int,
    block_size: int,
    samples_before_block: int,
    block_number: int,
) -> list[dict[str, int]]:
    events = []
    for index in range(nevents):
        sample, condition, code = struct.unpack_from(f"{prefix}BBH", header, 72 + index * 4)
        if sample >= block_size:
            raise ValueError(f"ERPSS event {index + 1} in block {block_number} lies outside the block")
        events.append(
            {
                "sample_offset": samples_before_block + int(sample) + 1,
                "event_code": int(code),
                "cond_code": int(condition),
            }
        )
    return events


def _microvolt_scale(metadata: dict[str, int]) -> float | None:
    ad_range_mv = metadata["ad_range_mv"]
    amplif = metadata["amplif"]
    ad_bits = metadata["ad_bits"]
    if ad_range_mv <= 0 or amplif <= 0 or not 1 <= ad_bits <= 32:
        return None
    return ad_range_mv * 1000.0 / amplif / (2**ad_bits)


def _sampling_rate(metadata: dict[str, int], sample_interval: int) -> float:
    clock_freq = metadata["clock_freq"]
    divider = metadata["divider"]
    nsteps = metadata["nsteps"]
    if clock_freq > 0 and divider > 0 and nsteps > 0:
        candidate = clock_freq / divider / nsteps
        return float(candidate) if np.isfinite(candidate) and candidate >= 0.5 else 0.0
    if sample_interval > 0:
        return 1_000_000.0 / sample_interval
    return 0.0


def _decompress(payload: bytes, sample_count: int, nchans: int, byteorder: str) -> np.ndarray:
    if len(payload) % 2:
        raise ValueError("ERPSS compressed data must contain complete 16-bit words")
    words = np.frombuffer(payload, dtype="<u2" if byteorder == "little" else ">u2")
    reader = _BitReader(words)
    output = np.empty(sample_count, dtype=np.int32)
    try:
        for index in range(sample_count):
            code = reader.read(4)
            absolute = False
            if code & 0b1000 == 0:
                value = _signed(code, 3)
            elif code & 0b0100 == 0:
                value = ((code & 1) << 4) | reader.read(4)
                if code & 0b0010:
                    value -= 32
            elif code & 0b0010 == 0:
                value = reader.read(8)
                if code & 1:
                    value -= 256
            else:
                value = _signed(reader.read(12), 12)
                absolute = True
            if not absolute:
                if index < nchans:
                    raise ValueError("ERPSS compressed channel starts with a delta instead of an absolute sample")
                value += int(output[index - nchans])
            if not -32768 <= value <= 32767:
                raise ValueError("ERPSS decompression produced a sample outside the int16 range")
            output[index] = value
    except EOFError as error:
        raise ValueError("ERPSS compressed data ended before all samples were decoded") from error
    return output.astype(np.int16)


def _signed(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


class _BitReader:
    def __init__(self, words: np.ndarray) -> None:
        self._words = words
        self._word_index = 0
        self._remaining = 0
        self._word = 0

    def read(self, count: int) -> int:
        value = 0
        while count:
            if self._remaining == 0:
                if self._word_index >= self._words.size:
                    raise EOFError
                self._word = int(self._words[self._word_index])
                self._word_index += 1
                self._remaining = 16
            take = min(count, self._remaining)
            shift = self._remaining - take
            value = (value << take) | ((self._word >> shift) & ((1 << take) - 1))
            self._remaining -= take
            count -= take
        return value


__all__ = ["read_erpss"]
