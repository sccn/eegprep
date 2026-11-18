"""Sun fdlibm ``exp`` and ``pow``.

Copyright (C) 1993, 2004 by Sun Microsystems, Inc.
Permission to use, copy, modify, and distribute this software is freely
granted, provided that this notice is preserved.
"""

from __future__ import annotations

import numpy as np


def _bits(high: int, low: int = 0) -> np.float64:
    return np.uint64((high << 32) | low).view(np.float64)


_ONE = np.float64(1.0)
_TWO = np.float64(2.0)
_HUGE = np.float64(1.0e300)
_TINY = np.float64(1.0e-300)
_TWO54 = _bits(0x43500000)
_TWOM54 = _bits(0x3C900000)
_TWOM1000 = _bits(0x01700000)
_EXP1 = np.float64(2.718281828459045235360)
_O_THRESHOLD = _bits(0x40862E42, 0xFEFA39EF)
_U_THRESHOLD = _bits(0xC0874910, 0xD52D3051)
_LN2_HI = _bits(0x3FE62E42, 0xFEE00000)
_LN2_LO = _bits(0x3DEA39EF, 0x35793C76)
_INVLN2 = _bits(0x3FF71547, 0x652B82FE)
_P1 = _bits(0x3FC55555, 0x5555553E)
_P2 = _bits(0xBF66C16C, 0x16BEBD93)
_P3 = _bits(0x3F11566A, 0xAF25DE2C)
_P4 = _bits(0xBEBBBD41, 0xC5D26BF1)
_P5 = _bits(0x3E663769, 0x72BEA4D0)
_BP = (_ONE, np.float64(1.5))
_DP_H = (np.float64(0.0), _bits(0x3FE2B803, 0x40000000))
_DP_L = (np.float64(0.0), _bits(0x3E4CFDEB, 0x43CFD006))
_L1 = _bits(0x3FE33333, 0x33333303)
_L2 = _bits(0x3FDB6DB6, 0xDB6FABFF)
_L3 = _bits(0x3FD55555, 0x518F264D)
_L4 = _bits(0x3FD17460, 0xA91D4101)
_L5 = _bits(0x3FCD864A, 0x93C9DB65)
_L6 = _bits(0x3FCA7E28, 0x4A454EEF)
_LG2 = _bits(0x3FE62E42, 0xFEFA39EF)
_LG2_H = _bits(0x3FE62E43, 0x00000000)
_LG2_L = _bits(0xBE205C61, 0x0CA86C39)
_OVT = _bits(0x3C971547, 0x652B82FE)
_CP = _bits(0x3FEEC709, 0xDC3A03FD)
_CP_H = _bits(0x3FEEC709, 0xE0000000)
_CP_L = _bits(0xBE3E2FE0, 0x145B01F5)
_IVLN2 = _bits(0x3FF71547, 0x652B82FE)
_IVLN2_H = _bits(0x3FF71547, 0x60000000)
_IVLN2_L = _bits(0x3E54AE0B, 0xF85DDF44)
_POW_2_1023 = _bits(0x7FE00000)


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
            return np.float64(_HUGE * _HUGE)
        if number < _U_THRESHOLD:
            return np.float64(_TWOM1000 * _TWOM1000)
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


def exp(values: np.ndarray | float) -> np.ndarray:
    """Return fdlibm ``exp`` of each value."""
    array = np.asarray(values, dtype=np.float64)
    output = np.empty(array.shape, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        for index in np.ndindex(array.shape):
            output[index] = _exp_scalar(array[index])
    return output


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
