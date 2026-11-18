"""Sun fdlibm ``exp`` and ``pow``.

Copyright (C) 1993, 2004 by Sun Microsystems, Inc.
Permission to use, copy, modify, and distribute this software is freely
granted, provided that this notice is preserved.
"""

from __future__ import annotations

import numpy as np


_ONE = np.float64(1.0)
_TWO = np.float64(2.0)
_HUGE = np.float64(1.0e300)
_TINY = np.float64(1.0e-300)
_TWO54 = np.float64.fromhex("0x1.0000000000000p+54")
_TWOM54 = np.float64.fromhex("0x1.0000000000000p-54")
_TWOM1000 = np.float64.fromhex("0x1.0000000000000p-1000")
_EXP1 = np.float64.fromhex("0x1.5bf0a8b145769p+1")
_O_THRESHOLD = np.float64.fromhex("0x1.62e42fefa39efp+9")
_U_THRESHOLD = np.float64.fromhex("-0x1.74910d52d3051p+9")
_LN2_HI = np.float64.fromhex("0x1.62e42fee00000p-1")
_LN2_LO = np.float64.fromhex("0x1.a39ef35793c76p-33")
_INVLN2 = np.float64.fromhex("0x1.71547652b82fep+0")
_P1 = np.float64.fromhex("0x1.555555555553ep-3")
_P2 = np.float64.fromhex("-0x1.6c16c16bebd93p-9")
_P3 = np.float64.fromhex("0x1.1566aaf25de2cp-14")
_P4 = np.float64.fromhex("-0x1.bbd41c5d26bf1p-20")
_P5 = np.float64.fromhex("0x1.6376972bea4d0p-25")
_BP = (_ONE, np.float64(1.5))
_DP_H = (np.float64(0.0), np.float64.fromhex("0x1.2b80340000000p-1"))
_DP_L = (np.float64(0.0), np.float64.fromhex("0x1.cfdeb43cfd006p-27"))
_L1 = np.float64.fromhex("0x1.3333333333303p-1")
_L2 = np.float64.fromhex("0x1.b6db6db6fabffp-2")
_L3 = np.float64.fromhex("0x1.55555518f264dp-2")
_L4 = np.float64.fromhex("0x1.17460a91d4101p-2")
_L5 = np.float64.fromhex("0x1.d864a93c9db65p-3")
_L6 = np.float64.fromhex("0x1.a7e284a454eefp-3")
_LG2 = np.float64.fromhex("0x1.62e42fefa39efp-1")
_LG2_H = np.float64.fromhex("0x1.62e4300000000p-1")
_LG2_L = np.float64.fromhex("-0x1.05c610ca86c39p-29")
_OVT = np.float64.fromhex("0x1.71547652b82fep-54")
_CP = np.float64.fromhex("0x1.ec709dc3a03fdp-1")
_CP_H = np.float64.fromhex("0x1.ec709e0000000p-1")
_CP_L = np.float64.fromhex("-0x1.e2fe0145b01f5p-28")
_IVLN2 = np.float64.fromhex("0x1.71547652b82fep+0")
_IVLN2_H = np.float64.fromhex("0x1.7154760000000p+0")
_IVLN2_L = np.float64.fromhex("0x1.4ae0bf85ddf44p-26")
_POW_2_1023 = np.float64.fromhex("0x1.0000000000000p+1023")
_EXP_BLOCK = 16384
with np.errstate(over="ignore", under="ignore"):
    _OVERFLOW = np.float64(_HUGE * _HUGE)
    _UNDERFLOW = np.float64(_TWOM1000 * _TWOM1000)


def _u32(value: int) -> int:
    return value & 0xFFFFFFFF


def _i32(value: int) -> int:
    value = _u32(value)
    if value >= 0x80000000:
        return value - 0x100000000
    return value


def _words(value: float) -> tuple[int, int]:
    bits = int(np.float64(value).view(np.uint64))
    return (bits >> 32) & 0xFFFFFFFF, bits & 0xFFFFFFFF


def _pack(high: int, low: int) -> np.float64:
    return np.uint64((_u32(high) << 32) | _u32(low)).view(np.float64)


def _high(value: float) -> int:
    return _words(value)[0]


def _low(value: float) -> int:
    return _words(value)[1]


def _set_high(value: float, high: int) -> np.float64:
    return _pack(high, _low(value))


def _set_low(value: float, low: int) -> np.float64:
    return _pack(_high(value), low)


def _scalbn(value: float, power: int) -> np.float64:
    number = np.float64(value)
    high, low = _words(number)
    exponent = (high & 0x7FF00000) >> 20
    if exponent == 0:
        if (low | (high & 0x7FFFFFFF)) == 0:
            return number
        number = np.float64(number * _TWO54)
        high = _high(number)
        exponent = ((high & 0x7FF00000) >> 20) - 54
        if power < -50000:
            return np.float64(_TINY * number)
    if exponent == 0x7FF:
        return np.float64(number + number)
    exponent += power
    if exponent > 0x7FE:
        return np.float64(_HUGE * np.copysign(_HUGE, number))
    if exponent > 0:
        return _set_high(number, (high & 0x800FFFFF) | (exponent << 20))
    if exponent <= -54:
        if power > 50000:
            return np.float64(_HUGE * np.copysign(_HUGE, number))
        return np.float64(_TINY * np.copysign(_TINY, number))
    exponent += 54
    return np.float64(_set_high(number, (high & 0x800FFFFF) | (exponent << 20)) * _TWOM54)


def _exp_scalar(value: float) -> np.float64:
    number = np.float64(value)
    high = _high(number)
    sign = (high >> 31) & 1
    high &= 0x7FFFFFFF
    if high >= 0x40862E42:
        if high >= 0x7FF00000:
            if ((high & 0xFFFFF) | _low(number)) != 0:
                return np.float64(number + number)
            return number if sign == 0 else np.float64(0.0)
        if number > _O_THRESHOLD:
            return _OVERFLOW
        if number < _U_THRESHOLD:
            return _UNDERFLOW
    if number == 1.0:
        return _EXP1

    reduced = number
    exponent = 0
    high_part = np.float64(0.0)
    low_part = np.float64(0.0)
    if high > 0x3FD62E42:
        if high < 0x3FF0A2B2:
            high_part = np.float64(number - (sign and -_LN2_HI or _LN2_HI))
            low_part = -_LN2_LO if sign else _LN2_LO
            exponent = 1 - sign - sign
        else:
            exponent = int(_INVLN2 * number + (np.float64(-0.5) if sign else np.float64(0.5)))
            scale = np.float64(exponent)
            high_part = np.float64(number - scale * _LN2_HI)
            low_part = np.float64(scale * _LN2_LO)
        reduced = np.float64(high_part - low_part)
    elif high < 0x3E300000:
        if _HUGE + number > _ONE:
            return np.float64(_ONE + number)
    square = np.float64(reduced * reduced)
    if exponent >= -1021:
        scale_word = _u32(0x3FF00000 + _i32(exponent << 20))
    else:
        scale_word = _u32(0x3FF00000 + _i32((exponent + 1000) << 20))
    twopk = _pack(scale_word, 0)
    correction = np.float64(reduced - square * (_P1 + square * (_P2 + square * (_P3 + square * (_P4 + square * _P5)))))
    if exponent == 0:
        return np.float64(_ONE - ((reduced * correction) / (correction - 2.0) - reduced))
    result = np.float64(_ONE - ((low_part - (reduced * correction) / (2.0 - correction)) - high_part))
    if exponent >= -1021:
        if exponent == 1024:
            return np.float64(result * 2.0 * _POW_2_1023)
        return np.float64(result * twopk)
    return np.float64(result * twopk * _TWOM1000)


def _pow_scalar(base: float, exponent: float) -> np.float64:
    number = np.float64(base)
    power = np.float64(exponent)
    high_x = _i32(_high(number))
    low_x = _low(number)
    high_y = _i32(_high(power))
    low_y = _low(power)
    abs_x = high_x & 0x7FFFFFFF
    abs_y = high_y & 0x7FFFFFFF

    if (abs_y | low_y) == 0:
        return _ONE
    if high_x == 0x3FF00000 and low_x == 0:
        return _ONE
    if (
        abs_x > 0x7FF00000
        or (abs_x == 0x7FF00000 and low_x != 0)
        or abs_y > 0x7FF00000
        or (abs_y == 0x7FF00000 and low_y != 0)
    ):
        return np.float64((number + 0.0) + (power + 0.0))

    integer = 0
    if high_x < 0:
        if abs_y >= 0x43400000:
            integer = 2
        elif abs_y >= 0x3FF00000:
            bits = (abs_y >> 20) - 0x3FF
            if bits > 20:
                shifted = low_y >> (52 - bits)
                if (shifted << (52 - bits)) == low_y:
                    integer = 2 - (shifted & 1)
            elif low_y == 0:
                shifted = abs_y >> (20 - bits)
                if (shifted << (20 - bits)) == abs_y:
                    integer = 2 - (shifted & 1)

    if low_y == 0:
        if abs_y == 0x7FF00000:
            if ((abs_x - 0x3FF00000) | low_x) == 0:
                return _ONE
            if abs_x >= 0x3FF00000:
                return power if high_y >= 0 else np.float64(0.0)
            return np.float64(-power) if high_y < 0 else np.float64(0.0)
        if abs_y == 0x3FF00000:
            return np.float64(_ONE / number) if high_y < 0 else number
        if high_y == 0x40000000:
            return np.float64(number * number)
        if high_y == 0x40080000:
            return np.float64(number * number * number)
        if high_y == 0x40100000:
            square = np.float64(number * number)
            return np.float64(square * square)
        if high_y == 0x3FE00000 and high_x >= 0:
            return np.float64(np.sqrt(number))

    magnitude = np.float64(np.abs(number))
    if low_x == 0 and abs_x in (0x7FF00000, 0, 0x3FF00000):
        result = magnitude
        if high_y < 0:
            result = np.float64(_ONE / result)
        if high_x < 0:
            if ((abs_x - 0x3FF00000) | integer) == 0:
                result = np.float64((result - result) / (result - result))
            elif integer == 1:
                result = np.float64(-result)
        return result

    negative = ((_u32(high_x) >> 31) - 1) | integer
    if negative == 0:
        return np.float64((number - number) / (number - number))
    sign = np.float64(-1.0) if (((_u32(high_x) >> 31) - 1) | (integer - 1)) == 0 else _ONE

    if abs_y > 0x41E00000:
        if abs_y > 0x43F00000:
            if abs_x <= 0x3FEFFFFF:
                return np.float64(_HUGE * _HUGE) if high_y < 0 else np.float64(_TINY * _TINY)
            if abs_x >= 0x3FF00000:
                return np.float64(_HUGE * _HUGE) if high_y > 0 else np.float64(_TINY * _TINY)
        if abs_x < 0x3FEFFFFF:
            overflow = np.float64(_HUGE * _HUGE) if high_y < 0 else np.float64(_TINY * _TINY)
            return np.float64(sign * overflow)
        if abs_x > 0x3FF00000:
            overflow = np.float64(_HUGE * _HUGE) if high_y > 0 else np.float64(_TINY * _TINY)
            return np.float64(sign * overflow)
        gap = np.float64(magnitude - _ONE)
        remainder = np.float64((gap * gap) * (0.5 - gap * (0.3333333333333333333333 - gap * 0.25)))
        upper = np.float64(_IVLN2_H * gap)
        lower = np.float64(gap * _IVLN2_L - remainder * _IVLN2)
        log_high = _set_low(np.float64(upper + lower), 0)
        log_low = np.float64(lower - (log_high - upper))
    else:
        exponent_bias = 0
        if abs_x < 0x00100000:
            magnitude = np.float64(magnitude * np.float64.fromhex("0x1.0p53"))
            exponent_bias -= 53
            abs_x = _high(magnitude)
        exponent_bias += (abs_x >> 20) - 0x3FF
        fraction = abs_x & 0x000FFFFF
        normalized = fraction | 0x3FF00000
        if fraction <= 0x3988E:
            interval = 0
        elif fraction < 0xBB67A:
            interval = 1
        else:
            interval = 0
            exponent_bias += 1
            normalized -= 0x00100000
        magnitude = _set_high(magnitude, normalized)
        numerator = np.float64(magnitude - _BP[interval])
        denominator = np.float64(_ONE / (magnitude + _BP[interval]))
        ratio = np.float64(numerator * denominator)
        ratio_high = _set_low(ratio, 0)
        split_high = _pack(((normalized >> 1) | 0x20000000) + 0x00080000 + (interval << 18), 0)
        split_low = np.float64(magnitude - (split_high - _BP[interval]))
        ratio_low = np.float64(denominator * ((numerator - ratio_high * split_high) - ratio_high * split_low))
        ratio_square = np.float64(ratio * ratio)
        series = np.float64(
            ratio_square
            * ratio_square
            * (
                _L1
                + ratio_square
                * (_L2 + ratio_square * (_L3 + ratio_square * (_L4 + ratio_square * (_L5 + ratio_square * _L6))))
            )
        )
        series = np.float64(series + ratio_low * (ratio_high + ratio))
        ratio_square = np.float64(ratio_high * ratio_high)
        poly_high = _set_low(np.float64(3.0 + ratio_square + series), 0)
        poly_low = np.float64(series - ((poly_high - 3.0) - ratio_square))
        product_high = np.float64(ratio_high * poly_high)
        product_low = np.float64(ratio_low * poly_high + poly_low * ratio)
        mixed_high = _set_low(np.float64(product_high + product_low), 0)
        mixed_low = np.float64(product_low - (mixed_high - product_high))
        log_high_full = np.float64(_CP_H * mixed_high)
        log_low_full = np.float64(_CP_L * mixed_high + mixed_low * _CP + _DP_L[interval])
        bias = np.float64(exponent_bias)
        log_high = _set_low(np.float64(((log_high_full + log_low_full) + _DP_H[interval]) + bias), 0)
        log_low = np.float64(log_low_full - (((log_high - bias) - _DP_H[interval]) - log_high_full))

    exponent_high = _set_low(power, 0)
    product_low = np.float64((power - exponent_high) * log_high + power * log_low)
    product_high = np.float64(exponent_high * log_high)
    combined = np.float64(product_low + product_high)
    combined_high, combined_low = _words(combined)
    combined_high_signed = _i32(combined_high)
    if combined_high_signed >= 0x40900000:
        if ((combined_high_signed - 0x40900000) | _i32(combined_low)) != 0:
            return np.float64(sign * _HUGE * _HUGE)
        if product_low + _OVT > combined - product_high:
            return np.float64(sign * _HUGE * _HUGE)
    elif (_u32(combined_high_signed) & 0x7FFFFFFF) >= 0x4090CC00:
        if (_i32(combined_high_signed - 0xC090CC00) | _i32(combined_low)) != 0:
            return np.float64(sign * _TINY * _TINY)
        if product_low <= combined - product_high:
            return np.float64(sign * _TINY * _TINY)

    absolute = combined_high_signed & 0x7FFFFFFF
    reduction = (absolute >> 20) - 0x3FF
    scale = 0
    if absolute > 0x3FE00000:
        rounded = _i32(combined_high_signed + (0x00100000 >> (reduction + 1)))
        reduction = ((_u32(rounded) & 0x7FFFFFFF) >> 20) - 0x3FF
        cutoff = _pack(rounded & ~_u32(0x000FFFFF >> reduction), 0)
        scale = ((_u32(rounded) & 0x000FFFFF) | 0x00100000) >> (20 - reduction)
        if combined_high_signed < 0:
            scale = -scale
        product_high = np.float64(product_high - cutoff)
    reduced_sum = _set_low(np.float64(product_low + product_high), 0)
    upper = np.float64(reduced_sum * _LG2_H)
    lower = np.float64((product_low - (reduced_sum - product_high)) * _LG2 + reduced_sum * _LG2_L)
    total = np.float64(upper + lower)
    tail = np.float64(lower - (total - upper))
    square = np.float64(total * total)
    series = np.float64(total - square * (_P1 + square * (_P2 + square * (_P3 + square * (_P4 + square * _P5)))))
    remainder = np.float64((total * series) / (series - _TWO) - (tail + total * tail))
    result = np.float64(_ONE - (remainder - total))
    result_high = _i32(_high(result) + _i32(scale << 20))
    if (result_high >> 20) <= 0:
        result = _scalbn(result, scale)
    else:
        result = _set_high(result, result_high)
    return np.float64(sign * result)


def _exp_series(reduced: np.ndarray) -> np.ndarray:
    square = reduced * reduced
    polynomial = square * _P5
    polynomial += _P4
    polynomial *= square
    polynomial += _P3
    polynomial *= square
    polynomial += _P2
    polynomial *= square
    polynomial += _P1
    polynomial *= square
    correction = reduced - polynomial
    ratio = reduced * correction
    ratio /= correction - 2.0
    ratio -= reduced
    return _ONE - ratio


def _twopk(exponent: np.ndarray) -> np.ndarray:
    shifted = exponent.astype(np.int64) << np.int64(20)
    low = shifted & np.int64(0xFFFFFFFF)
    signed = np.where(low >= np.int64(0x80000000), low - np.int64(0x100000000), low)
    scale_word = (np.int64(0x3FF00000) + signed) & np.int64(0xFFFFFFFF)
    return (scale_word.astype(np.uint64) << np.uint64(32)).view(np.float64)


def exp(values: np.ndarray | float) -> np.ndarray:
    """Return fdlibm ``exp`` of each value."""
    requested = np.asarray(values, dtype=np.float64)
    number = np.ascontiguousarray(requested)
    bits = number.view(np.uint64)
    high_raw = (bits >> np.uint64(32)).astype(np.uint32)
    sign = high_raw >> np.uint32(31)
    high = high_raw & np.uint32(0x7FFFFFFF)
    low = bits.astype(np.uint32)
    if number.size and int(high.max()) <= 0x3FD62E42:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            flat = number.ravel()
            high_flat = high.ravel()
            series = np.empty(flat.shape, dtype=np.float64)
            for start in range(0, flat.size, _EXP_BLOCK):
                stop = start + _EXP_BLOCK
                block = flat[start:stop]
                block_series = _exp_series(block)
                small = high_flat[start:stop] < np.uint32(0x3E300000)
                if np.any(small):
                    block_series = np.where(small, _ONE + block, block_series)
                series[start:stop] = block_series
        return series.reshape(requested.shape)

    nonfinite = high >= np.uint32(0x7FF00000)
    nan_bits = ((high & np.uint32(0xFFFFF)).astype(np.uint64) | low.astype(np.uint64)) != 0
    large = high >= np.uint32(0x40862E42)
    output = np.empty(number.shape, dtype=np.float64)
    done = np.zeros(number.shape, dtype=bool)

    nan = nonfinite & nan_bits
    positive_inf = nonfinite & ~nan_bits & (sign == 0)
    negative_inf = nonfinite & ~nan_bits & (sign == 1)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        output[nan] = number[nan] + number[nan]
    output[positive_inf] = number[positive_inf]
    output[negative_inf] = 0.0
    done |= nonfinite

    overflow = large & ~done & (number > _O_THRESHOLD)
    underflow = large & ~done & (number < _U_THRESHOLD)
    output[overflow] = _OVERFLOW
    output[underflow] = _UNDERFLOW
    done |= overflow | underflow

    one = ~done & (number == 1.0)
    output[one] = _EXP1
    done |= one

    small = ~done & (high < np.uint32(0x3E300000))
    output[small] = _ONE + number[small]
    done |= small

    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        negative = sign == 1
        mid = ~done & (high > np.uint32(0x3FD62E42)) & (high < np.uint32(0x3FF0A2B2))
        wide = ~done & (high > np.uint32(0x3FD62E42)) & ~mid
        high_part = np.zeros(number.shape, dtype=np.float64)
        low_part = np.zeros(number.shape, dtype=np.float64)
        exponent = np.zeros(number.shape, dtype=np.int64)
        high_part = np.where(mid, number - np.where(negative, -_LN2_HI, _LN2_HI), high_part)
        low_part = np.where(mid, np.where(negative, -_LN2_LO, _LN2_LO), low_part)
        exponent = np.where(mid, np.where(negative, np.int64(-1), np.int64(1)), exponent)
        biased = np.where(wide, _INVLN2 * number + np.where(negative, np.float64(-0.5), np.float64(0.5)), 0.0)
        reduced_exponent = np.trunc(biased).astype(np.int64)
        scale = reduced_exponent.astype(np.float64)
        high_part = np.where(wide, number - scale * _LN2_HI, high_part)
        low_part = np.where(wide, scale * _LN2_LO, low_part)
        exponent = np.where(wide, reduced_exponent, exponent)
        reduced = np.where(mid | wide, high_part - low_part, number)
        square = reduced * reduced
        correction = reduced - square * (_P1 + square * (_P2 + square * (_P3 + square * (_P4 + square * _P5))))
        zero_scale = _exp_series(reduced)
        shifted = _ONE - ((low_part - (reduced * correction) / (2.0 - correction)) - high_part)
        result = np.where(exponent == 0, zero_scale, shifted)
        twopk = _twopk(np.where(exponent >= -1021, exponent, exponent + 1000))
        scaled = np.where(exponent == 0, result, result * twopk)
        scaled = np.where(exponent == 1024, result * 2.0 * _POW_2_1023, scaled)
        scaled = np.where(exponent < -1021, result * twopk * _TWOM1000, scaled)
    return np.where(done, output, scaled).reshape(requested.shape)


def power(base: np.ndarray | float, exponent: np.ndarray | float) -> np.ndarray:
    """Return fdlibm ``x^y`` for each pair."""
    base_array = np.asarray(base, dtype=np.float64)
    exponent_array = np.asarray(exponent, dtype=np.float64)
    base_b, exponent_b = np.broadcast_arrays(base_array, exponent_array)
    output = np.empty(base_b.shape, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        for index in np.ndindex(output.shape):
            output[index] = _pow_scalar(base_b[index], exponent_b[index])
    return output
