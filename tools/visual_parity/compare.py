"""Compare EEGLAB and EEGPrep screenshots for visual parity review."""

from __future__ import annotations

import argparse
import binascii
import pathlib
import struct
import sys
import zlib
from dataclasses import dataclass


DEFAULT_OUTPUT_DIR = pathlib.Path(".visual-parity")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class ComparisonResult:
    """Image comparison metrics for one visual parity case."""

    reference_path: pathlib.Path
    candidate_path: pathlib.Path
    size_mismatch: bool
    mean_abs_delta: float
    max_abs_delta: float
    different_pixel_ratio: float


@dataclass(frozen=True)
class RgbaImage:
    """A minimal RGBA image container used to avoid runtime image dependencies."""

    width: int
    height: int
    pixels: bytes


def _iter_png_chunks(data: bytes):
    offset = len(PNG_SIGNATURE)
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        yield chunk_type, chunk_data
        offset += 12 + length


def _paeth(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    left_distance = abs(estimate - left)
    up_distance = abs(estimate - up)
    up_left_distance = abs(estimate - up_left)
    if left_distance <= up_distance and left_distance <= up_left_distance:
        return left
    if up_distance <= up_left_distance:
        return up
    return up_left


def _read_image(path: pathlib.Path) -> RgbaImage:
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"only PNG screenshots are supported: {path}")

    width = height = bit_depth = color_type = interlace = None
    compressed_parts: list[bytes] = []
    for chunk_type, chunk_data in _iter_png_chunks(data):
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _compression, _filter, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
        elif chunk_type == b"IDAT":
            compressed_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None or bit_depth is None or color_type is None:
        raise ValueError(f"invalid PNG file: {path}")
    if bit_depth != 8 or interlace != 0:
        raise ValueError(f"only 8-bit non-interlaced PNG screenshots are supported: {path}")

    channels_by_color_type = {0: 1, 2: 3, 4: 2, 6: 4}
    if color_type not in channels_by_color_type:
        raise ValueError(f"unsupported PNG color type {color_type}: {path}")

    channels = channels_by_color_type[color_type]
    bytes_per_pixel = channels
    row_length = width * channels
    raw = zlib.decompress(b"".join(compressed_parts))
    rows: list[bytearray] = []
    offset = 0
    previous = bytearray(row_length)
    for _row_index in range(height):
        filter_type = raw[offset]
        offset += 1
        row = bytearray(raw[offset : offset + row_length])
        offset += row_length
        for index in range(row_length):
            left = row[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            up = previous[index]
            up_left = previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            if filter_type == 1:
                row[index] = (row[index] + left) & 0xFF
            elif filter_type == 2:
                row[index] = (row[index] + up) & 0xFF
            elif filter_type == 3:
                row[index] = (row[index] + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                row[index] = (row[index] + _paeth(left, up, up_left)) & 0xFF
            elif filter_type != 0:
                raise ValueError(f"unsupported PNG filter {filter_type}: {path}")
        rows.append(row)
        previous = row

    rgba = bytearray(width * height * 4)
    output = 0
    for row in rows:
        input_index = 0
        for _column in range(width):
            if color_type == 0:
                value = row[input_index]
                rgba[output : output + 4] = bytes((value, value, value, 255))
                input_index += 1
            elif color_type == 2:
                rgba[output : output + 4] = bytes((row[input_index], row[input_index + 1], row[input_index + 2], 255))
                input_index += 3
            elif color_type == 4:
                value = row[input_index]
                rgba[output : output + 4] = bytes((value, value, value, row[input_index + 1]))
                input_index += 2
            else:
                rgba[output : output + 4] = row[input_index : input_index + 4]
                input_index += 4
            output += 4
    return RgbaImage(width=width, height=height, pixels=bytes(rgba))


def _png_chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
    checksum = binascii.crc32(chunk_type)
    checksum = binascii.crc32(chunk_data, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(chunk_data)) + chunk_type + chunk_data + struct.pack(">I", checksum)


def _save_image(path: pathlib.Path, image: RgbaImage) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_length = image.width * 4
    raw = bytearray()
    for row_index in range(image.height):
        start = row_index * row_length
        raw.append(0)
        raw.extend(image.pixels[start : start + row_length])
    ihdr = struct.pack(">IIBBBBB", image.width, image.height, 8, 6, 0, 0, 0)
    png = PNG_SIGNATURE + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(bytes(raw))) + _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def _pad_to_height(image: RgbaImage, height: int) -> RgbaImage:
    if image.height == height:
        return image
    row_length = image.width * 4
    padding = bytes((255, 255, 255, 255)) * image.width * (height - image.height)
    return RgbaImage(width=image.width, height=height, pixels=image.pixels + padding)


def write_side_by_side(reference: RgbaImage, candidate: RgbaImage, path: pathlib.Path) -> None:
    """Write a side-by-side reference/candidate image."""
    height = max(reference.height, candidate.height)
    reference = _pad_to_height(reference, height)
    candidate = _pad_to_height(candidate, height)
    output_width = reference.width + 16 + candidate.width
    output = bytearray(bytes((255, 255, 255, 255)) * output_width * height)
    for row_index in range(height):
        out_row = row_index * output_width * 4
        ref_row = row_index * reference.width * 4
        cand_row = row_index * candidate.width * 4
        output[out_row : out_row + reference.width * 4] = reference.pixels[
            ref_row : ref_row + reference.width * 4
        ]
        cand_start = out_row + (reference.width + 16) * 4
        output[cand_start : cand_start + candidate.width * 4] = candidate.pixels[
            cand_row : cand_row + candidate.width * 4
        ]
    _save_image(path, RgbaImage(width=output_width, height=height, pixels=bytes(output)))


def compare_images(
    reference_path: pathlib.Path,
    candidate_path: pathlib.Path,
    diff_path: pathlib.Path | None = None,
    side_by_side_path: pathlib.Path | None = None,
    pixel_threshold: float = 0.02,
) -> ComparisonResult:
    """Compare two screenshots and optionally write visual artifacts."""
    reference = _read_image(reference_path)
    candidate = _read_image(candidate_path)
    size_mismatch = reference.width != candidate.width or reference.height != candidate.height

    height = min(reference.height, candidate.height)
    width = min(reference.width, candidate.width)
    total_delta = 0
    max_delta = 0
    different_pixels = 0
    diff_pixels = bytearray(width * height * 4)
    threshold_value = int(pixel_threshold * 255)
    for row_index in range(height):
        for column_index in range(width):
            ref_index = (row_index * reference.width + column_index) * 4
            cand_index = (row_index * candidate.width + column_index) * 4
            diff_index = (row_index * width + column_index) * 4
            red_delta = abs(reference.pixels[ref_index] - candidate.pixels[cand_index])
            green_delta = abs(reference.pixels[ref_index + 1] - candidate.pixels[cand_index + 1])
            blue_delta = abs(reference.pixels[ref_index + 2] - candidate.pixels[cand_index + 2])
            pixel_max_delta = max(red_delta, green_delta, blue_delta)
            total_delta += red_delta + green_delta + blue_delta
            max_delta = max(max_delta, pixel_max_delta)
            if pixel_max_delta > threshold_value:
                different_pixels += 1
            diff_pixels[diff_index : diff_index + 4] = bytes((red_delta, green_delta, blue_delta, 255))

    compared_pixels = max(width * height, 1)
    if diff_path is not None:
        _save_image(diff_path, RgbaImage(width=width, height=height, pixels=bytes(diff_pixels)))
    if side_by_side_path is not None:
        write_side_by_side(reference, candidate, side_by_side_path)

    if size_mismatch:
        mean_abs_delta = 1.0
        max_abs_delta = 1.0
        different_pixel_ratio = 1.0
    else:
        mean_abs_delta = total_delta / (compared_pixels * 3 * 255)
        max_abs_delta = max_delta / 255
        different_pixel_ratio = different_pixels / compared_pixels

    return ComparisonResult(
        reference_path=reference_path,
        candidate_path=candidate_path,
        size_mismatch=size_mismatch,
        mean_abs_delta=mean_abs_delta,
        max_abs_delta=max_abs_delta,
        different_pixel_ratio=different_pixel_ratio,
    )


def write_report(
    case_id: str,
    result: ComparisonResult,
    report_path: pathlib.Path,
    diff_path: pathlib.Path,
    side_by_side_path: pathlib.Path,
) -> None:
    """Write a Markdown report for human or VLM inspection."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                f"# Visual Parity Report: {case_id}",
                "",
                f"- Reference: `{result.reference_path}`",
                f"- Candidate: `{result.candidate_path}`",
                f"- Side by side: `{side_by_side_path}`",
                f"- Diff: `{diff_path}`",
                f"- Size mismatch: `{result.size_mismatch}`",
                f"- Mean absolute delta: `{result.mean_abs_delta:.6f}`",
                f"- Max absolute delta: `{result.max_abs_delta:.6f}`",
                f"- Different pixel ratio: `{result.different_pixel_ratio:.6f}`",
                "",
                "## VLM Review Prompt",
                "",
                "Compare the EEGLAB reference screenshot with the EEGPrep candidate screenshot from a user perspective.",
                "Prioritize menu labels, item order, enabled/disabled state, dialog layout, control labels, spacing, and obvious visual hierarchy.",
                "Ignore tiny antialiasing differences, OS chrome differences, and font rendering noise unless they change usability.",
                "Return specific actionable feedback for EEGPrep UI changes.",
                "",
            ]
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="Visual parity case id")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference", type=pathlib.Path, help="Reference screenshot path")
    parser.add_argument("--candidate", type=pathlib.Path, help="Candidate screenshot path")
    parser.add_argument("--pixel-threshold", type=float, default=0.02)
    parser.add_argument("--fail-threshold", type=float, help="Fail if mean absolute delta exceeds this value")
    args = parser.parse_args(argv)

    case_dir = args.output_dir / args.case
    reference_path = args.reference or case_dir / "eeglab.png"
    candidate_path = args.candidate or case_dir / "eegprep.png"
    diff_path = case_dir / "diff.png"
    side_by_side_path = case_dir / "side_by_side.png"
    report_path = case_dir / "report.md"

    for image_path in (reference_path, candidate_path):
        if not image_path.exists():
            parser.exit(1, f"missing screenshot: {image_path}\n")

    result = compare_images(
        reference_path,
        candidate_path,
        diff_path=diff_path,
        side_by_side_path=side_by_side_path,
        pixel_threshold=args.pixel_threshold,
    )
    write_report(args.case, result, report_path, diff_path, side_by_side_path)
    print(f"report: {report_path}")
    print(f"mean_abs_delta: {result.mean_abs_delta:.6f}")
    print(f"different_pixel_ratio: {result.different_pixel_ratio:.6f}")

    if args.fail_threshold is not None and (
        result.size_mismatch or result.mean_abs_delta > args.fail_threshold
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
