"""Sun fdlibm ``exp``, ``pow``, ``sin``, and ``cos``.

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


_S1 = _pack(0xBFC55555, 0x55555549)
_S2 = _pack(0x3F811111, 0x1110F8A6)
_S3 = _pack(0xBF2A01A0, 0x19C161D5)
_S4 = _pack(0x3EC71DE3, 0x57B1FE7D)
_S5 = _pack(0xBE5AE5E6, 0x8A2B9CEB)
_S6 = _pack(0x3DE5D93A, 0x5ACFD57C)
_C1 = _pack(0x3FA55555, 0x5555554C)
_C2 = _pack(0xBF56C16C, 0x16C15177)
_C3 = _pack(0x3EFA01A0, 0x19CB1590)
_C4 = _pack(0xBE927E4F, 0x809C52AD)
_C5 = _pack(0x3E21EE9E, 0xBDB4B1C4)
_C6 = _pack(0xBDA8FAE9, 0xBE8838D4)
_INVPIO2 = _pack(0x3FE45F30, 0x6DC9C883)
_PIO2_1 = _pack(0x3FF921FB, 0x54400000)
_PIO2_1T = _pack(0x3DD0B461, 0x1A626331)
_PIO2_2 = _pack(0x3DD0B461, 0x1A600000)
_PIO2_2T = _pack(0x3BA3198A, 0x2E037073)
_PIO2_3 = _pack(0x3BA3198A, 0x2E000000)
_PIO2_3T = _pack(0x397B839A, 0x252049C1)
_TWO24 = np.float64(16777216.0)
_TWON24 = _pack(0x3E700000, 0x00000000)
_PIO2_CHUNKS = (
    _pack(0x3FF921FB, 0x40000000),
    _pack(0x3E74442D, 0x00000000),
    _pack(0x3CF84698, 0x80000000),
    _pack(0x3B78CC51, 0x60000000),
    _pack(0x39F01B83, 0x80000000),
    _pack(0x387A2520, 0x40000000),
    _pack(0x36E38222, 0x80000000),
    _pack(0x3569F31D, 0x00000000),
)
_TWO_OVER_PI = (
    0xA2F983,
    0x6E4E44,
    0x1529FC,
    0x2757D1,
    0xF534DD,
    0xC0DB62,
    0x95993C,
    0x439041,
    0xFE5163,
    0xABDEBB,
    0xC561B7,
    0x246E3A,
    0x424DD2,
    0xE00649,
    0x2EEA09,
    0xD1921C,
    0xFE1DEB,
    0x1CB129,
    0xA73EE8,
    0x8235F5,
    0x2EBB44,
    0x84E99C,
    0x7026B4,
    0x5F7E41,
    0x3991D6,
    0x398353,
    0x39F49C,
    0x845F8B,
    0xBDF928,
    0x3B1FF8,
    0x97FFDE,
    0x05980F,
    0xEF2F11,
    0x8B5A0A,
    0x6D1F6D,
    0x367ECF,
    0x27CB09,
    0xB74F46,
    0x3F669E,
    0x5FEA2D,
    0x7527BA,
    0xC7EBE5,
    0xF17B3D,
    0x0739F7,
    0x8A5292,
    0xEA6BFB,
    0x5FB11F,
    0x8D5D08,
    0x560330,
    0x46FC7B,
    0x6BABF0,
    0xCFBC20,
    0x9AF436,
    0x1DA9E3,
    0x91615E,
    0xE61B08,
    0x659985,
    0x5F14A0,
    0x68408D,
    0xFFD880,
    0x4D7327,
    0x310606,
    0x1556CA,
    0x73A8C9,
    0x60E27B,
    0xC08C6B,
)
_NPIO2_HW = (
    0x3FF921FB,
    0x400921FB,
    0x4012D97C,
    0x401921FB,
    0x401F6A7A,
    0x4022D97C,
    0x4025FDBB,
    0x402921FB,
    0x402C463A,
    0x402F6A7A,
    0x4031475C,
    0x4032D97C,
    0x40346B9C,
    0x4035FDBB,
    0x40378FDB,
    0x403921FB,
    0x403AB41B,
    0x403C463A,
    0x403DD85A,
    0x403F6A7A,
    0x40407E4C,
    0x4041475C,
    0x4042106C,
    0x4042D97C,
    0x4043A28C,
    0x40446B9C,
    0x404534AC,
    0x4045FDBB,
    0x4046C6CB,
    0x40478FDB,
    0x404858EB,
    0x404921FB,
)
_TRIG_INIT_JK = (2, 3, 4, 6)
_NPIO2_HW_ARRAY = np.asarray(_NPIO2_HW, dtype=np.uint32)


def _trunc_int(value: float) -> int:
    return int(np.trunc(np.float64(value)))


def _kernel_sin(x_value: float, y_value: float, tail: int) -> np.float64:
    number = np.float64(x_value)
    tail_value = np.float64(y_value)
    if (_high(number) & 0x7FFFFFFF) < 0x3E400000:
        return number
    square = np.float64(number * number)
    cube = np.float64(square * number)
    remainder = np.float64(_S2 + square * (_S3 + square * (_S4 + square * (_S5 + square * _S6))))
    if tail == 0:
        return np.float64(number + cube * (_S1 + square * remainder))
    return np.float64(number - ((square * ((_ONE / 2.0) * tail_value - cube * remainder) - tail_value) - cube * _S1))


def _kernel_cos(x_value: float, y_value: float) -> np.float64:
    number = np.float64(x_value)
    tail_value = np.float64(y_value)
    absolute = _high(number) & 0x7FFFFFFF
    if absolute < 0x3E400000:
        return _ONE
    square = np.float64(number * number)
    remainder = np.float64(
        square * (_C1 + square * (_C2 + square * (_C3 + square * (_C4 + square * (_C5 + square * _C6)))))
    )
    if absolute < 0x3FD33333:
        return np.float64(_ONE - ((square / 2.0) - (square * remainder - number * tail_value)))
    if absolute > 0x3FE90000:
        quarter = np.float64(0.28125)
    else:
        quarter = _pack(absolute - 0x00200000, 0)
    high_square = np.float64(square / 2.0 - quarter)
    head = np.float64(_ONE - quarter)
    return np.float64(head - (high_square - (square * remainder - number * tail_value)))


def _kernel_rem_pio2(chunks: list[float], exponent: int, count: int) -> tuple[int, list[np.float64]]:
    terms = _TRIG_INIT_JK[2]
    kept = terms
    last = count - 1
    start = (exponent - 3) // 24
    if start < 0:
        start = 0
    scale = exponent - 24 * (start + 1)
    span = last + terms
    factors = [np.float64(0.0)] * (span + 8)
    offset = start - last
    for index in range(span + 1):
        table = offset + index
        factors[index] = np.float64(0.0) if table < 0 else np.float64(_TWO_OVER_PI[table])
    product = [np.float64(0.0)] * (span + 8)
    for index in range(terms + 1):
        total = np.float64(0.0)
        for piece in range(count):
            total = np.float64(total + np.float64(chunks[piece]) * factors[last + index - piece])
        product[index] = total
    used = terms
    while True:
        digits = [0] * (used + 8)
        remainder = product[used]
        for index, place in enumerate(range(used, 0, -1)):
            whole = np.float64(_trunc_int(np.float64(_TWON24 * remainder)))
            digits[index] = _trunc_int(np.float64(remainder - _TWO24 * whole))
            remainder = np.float64(product[place - 1] + whole)
        scaled = _scalbn(remainder, scale)
        trimmed = np.float64(scaled - np.float64(8.0 * np.floor(np.float64(scaled * 0.125))))
        cycles = _trunc_int(trimmed)
        fraction = np.float64(trimmed - np.float64(cycles))
        indicator = 0
        if scale > 0:
            shift = 24 - scale
            extra = digits[used - 1] >> shift
            cycles += extra
            digits[used - 1] -= extra << shift
            indicator = digits[used - 1] >> (23 - scale)
        elif scale == 0:
            indicator = digits[used - 1] >> 23
        elif fraction >= np.float64(0.5):
            indicator = 2
        if indicator > 0:
            cycles += 1
            carry = 0
            for index in range(used):
                digit = digits[index]
                if carry == 0:
                    if digit != 0:
                        carry = 1
                        digits[index] = 0x1000000 - digit
                else:
                    digits[index] = 0xFFFFFF - digit
            if scale > 0:
                if scale == 1:
                    digits[used - 1] &= 0x7FFFFF
                elif scale == 2:
                    digits[used - 1] &= 0x3FFFFF
            if indicator == 2:
                fraction = np.float64(_ONE - fraction)
                if carry != 0:
                    fraction = np.float64(fraction - _scalbn(_ONE, scale))
        if fraction == 0.0:
            mixed = 0
            for index in range(used - 1, kept - 1, -1):
                mixed |= digits[index]
            if mixed == 0:
                extra_terms = 1
                while digits[kept - extra_terms] == 0:
                    extra_terms += 1
                for index in range(used + 1, used + extra_terms + 1):
                    factors[last + index] = np.float64(_TWO_OVER_PI[start + index])
                    total = np.float64(0.0)
                    for piece in range(count):
                        total = np.float64(total + np.float64(chunks[piece]) * factors[last + index - piece])
                    product[index] = total
                used += extra_terms
                continue
        break
    if fraction == 0.0:
        used -= 1
        scale -= 24
        while digits[used] == 0:
            used -= 1
            scale -= 24
    else:
        fraction = _scalbn(fraction, -scale)
        if fraction >= _TWO24:
            whole = np.float64(_trunc_int(np.float64(_TWON24 * fraction)))
            digits[used] = _trunc_int(np.float64(fraction - _TWO24 * whole))
            used += 1
            scale += 24
            digits[used] = _trunc_int(whole)
        else:
            digits[used] = _trunc_int(fraction)
    weight = _scalbn(_ONE, scale)
    for index in range(used, -1, -1):
        product[index] = np.float64(weight * np.float64(digits[index]))
        weight = np.float64(weight * _TWON24)
    compressed = [np.float64(0.0)] * (used + 1)
    for index in range(used, -1, -1):
        total = np.float64(0.0)
        for piece in range(min(kept, used - index) + 1):
            total = np.float64(total + _PIO2_CHUNKS[piece] * product[index + piece])
        compressed[used - index] = total
    total = np.float64(0.0)
    for index in range(used, -1, -1):
        total = np.float64(total + compressed[index])
    primary = total if indicator == 0 else np.float64(-total)
    correction = np.float64(compressed[0] - total)
    for index in range(1, used + 1):
        correction = np.float64(correction + compressed[index])
    secondary = correction if indicator == 0 else np.float64(-correction)
    return cycles & 7, [primary, secondary]


def _rem_pio2(value: float) -> tuple[int, np.float64, np.float64]:
    number = np.float64(value)
    high = _i32(_high(number))
    absolute = high & 0x7FFFFFFF
    if absolute <= 0x3FE921FB:
        return 0, number, np.float64(0.0)
    if absolute < 0x4002D97C:
        if high > 0:
            reduced = np.float64(number - _PIO2_1)
            if absolute != 0x3FF921FB:
                head = np.float64(reduced - _PIO2_1T)
                return 1, head, np.float64((reduced - head) - _PIO2_1T)
            reduced = np.float64(reduced - _PIO2_2)
            head = np.float64(reduced - _PIO2_2T)
            return 1, head, np.float64((reduced - head) - _PIO2_2T)
        reduced = np.float64(number + _PIO2_1)
        if absolute != 0x3FF921FB:
            head = np.float64(reduced + _PIO2_1T)
            return -1, head, np.float64((reduced - head) + _PIO2_1T)
        reduced = np.float64(reduced + _PIO2_2)
        head = np.float64(reduced + _PIO2_2T)
        return -1, head, np.float64((reduced - head) + _PIO2_2T)
    if absolute <= 0x413921FB:
        magnitude = np.float64(np.abs(number))
        cycles = _trunc_int(np.float64(magnitude * _INVPIO2 + _ONE / 2.0))
        scale = np.float64(cycles)
        remainder = np.float64(magnitude - scale * _PIO2_1)
        tail = np.float64(scale * _PIO2_1T)
        if cycles < 32 and absolute != _NPIO2_HW[cycles - 1]:
            head = np.float64(remainder - tail)
        else:
            exponent = absolute >> 20
            head = np.float64(remainder - tail)
            gap = exponent - ((_high(head) >> 20) & 0x7FF)
            if gap > 16:
                saved = remainder
                second = np.float64(scale * _PIO2_2)
                remainder = np.float64(saved - second)
                tail = np.float64(scale * _PIO2_2T - ((saved - remainder) - second))
                head = np.float64(remainder - tail)
                gap = exponent - ((_high(head) >> 20) & 0x7FF)
                if gap > 49:
                    saved = remainder
                    third = np.float64(scale * _PIO2_3)
                    remainder = np.float64(saved - third)
                    tail = np.float64(scale * _PIO2_3T - ((saved - remainder) - third))
                    head = np.float64(remainder - tail)
        low = np.float64((remainder - head) - tail)
        if high < 0:
            return -cycles, np.float64(-head), np.float64(-low)
        return cycles, head, low
    if absolute >= 0x7FF00000:
        nan = np.float64(number - number)
        return 0, nan, nan
    magnitude = np.float64(np.abs(number))
    shift = (absolute >> 20) - 1046
    scaled = _set_high(magnitude, absolute - (shift << 20))
    pieces = [np.float64(0.0), np.float64(0.0), np.float64(0.0)]
    for index in range(2):
        pieces[index] = np.float64(_trunc_int(scaled))
        scaled = np.float64((scaled - pieces[index]) * _TWO24)
    pieces[2] = scaled
    count = 3
    while pieces[count - 1] == 0.0:
        count -= 1
    cycles, reduced = _kernel_rem_pio2(pieces, shift, count)
    if high < 0:
        return -cycles, np.float64(-reduced[0]), np.float64(-reduced[1])
    return cycles, reduced[0], reduced[1]


def _sin_scalar(value: float) -> np.float64:
    number = np.float64(value)
    absolute = _high(number) & 0x7FFFFFFF
    if absolute <= 0x3FE921FB:
        return _kernel_sin(number, 0.0, 0)
    if absolute >= 0x7FF00000:
        return np.float64(number - number)
    cycles, head, tail = _rem_pio2(number)
    quadrant = cycles & 3
    if quadrant == 0:
        return _kernel_sin(head, tail, 1)
    if quadrant == 1:
        return _kernel_cos(head, tail)
    if quadrant == 2:
        return np.float64(-_kernel_sin(head, tail, 1))
    return np.float64(-_kernel_cos(head, tail))


def _cos_scalar(value: float) -> np.float64:
    number = np.float64(value)
    absolute = _high(number) & 0x7FFFFFFF
    if absolute <= 0x3FE921FB:
        return _kernel_cos(number, 0.0)
    if absolute >= 0x7FF00000:
        return np.float64(number - number)
    cycles, head, tail = _rem_pio2(number)
    quadrant = cycles & 3
    if quadrant == 0:
        return _kernel_cos(head, tail)
    if quadrant == 1:
        return np.float64(-_kernel_sin(head, tail, 1))
    if quadrant == 2:
        return np.float64(-_kernel_cos(head, tail))
    return _kernel_sin(head, tail, 1)


_HALF = _ONE / np.float64(2.0)


def _unsigned_high(number: np.ndarray) -> np.ndarray:
    return (number.view(np.uint64) >> np.uint64(32)).astype(np.uint32)


def _exponent_field(number: np.ndarray) -> np.ndarray:
    return (_unsigned_high(number).astype(np.int64) >> np.int64(20)) & np.int64(0x7FF)


def _kernel_sin_array(number: np.ndarray, tail: np.ndarray, use_tail: np.ndarray | bool) -> np.ndarray:
    absolute = _unsigned_high(number) & np.uint32(0x7FFFFFFF)
    square = number * number
    cube = square * number
    remainder = _S2 + square * (_S3 + square * (_S4 + square * (_S5 + square * _S6)))
    plain = number + cube * (_S1 + square * remainder)
    adjusted = number - ((square * (_HALF * tail - cube * remainder) - tail) - cube * _S1)
    polynomial = np.where(use_tail, adjusted, plain)
    return np.where(absolute < np.uint32(0x3E400000), number, polynomial)


def _kernel_cos_array(number: np.ndarray, tail: np.ndarray) -> np.ndarray:
    absolute = _unsigned_high(number) & np.uint32(0x7FFFFFFF)
    square = number * number
    remainder = square * (_C1 + square * (_C2 + square * (_C3 + square * (_C4 + square * (_C5 + square * _C6)))))
    inner = square * remainder - number * tail
    half_square = square / np.float64(2.0)
    small_result = _ONE - (half_square - inner)
    wide = absolute > np.uint32(0x3FE90000)
    quarter_bits = (absolute.astype(np.uint64) - np.uint64(0x00200000)) << np.uint64(32)
    quarter = np.where(wide, np.float64(0.28125), quarter_bits.view(np.float64))
    wide_result = (_ONE - quarter) - ((half_square - quarter) - inner)
    polynomial = np.where(absolute < np.uint32(0x3FD33333), small_result, wide_result)
    return np.where(absolute < np.uint32(0x3E400000), _ONE, polynomial)


def _store_special(
    mask: np.ndarray,
    cycles: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    count: int,
    primary: np.ndarray,
    low: np.ndarray,
) -> None:
    cycles[mask] = count
    head[mask] = primary[mask]
    tail[mask] = low[mask]


def _reduce_array(number: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = number.reshape(-1)
    raw_high = _unsigned_high(flat)
    negative = raw_high >= np.uint32(0x80000000)
    absolute = raw_high & np.uint32(0x7FFFFFFF)
    cycles = np.zeros(flat.shape, dtype=np.int64)
    head = np.array(flat, dtype=np.float64, copy=True)
    tail = np.zeros(flat.shape, dtype=np.float64)
    use_tail = np.zeros(flat.shape, dtype=bool)

    special = (absolute > np.uint32(0x3FE921FB)) & (absolute < np.uint32(0x4002D97C))
    if np.any(special):
        values = flat[special]
        neg = negative[special]
        near = absolute[special] == np.uint32(0x3FF921FB)
        special_cycles = np.empty(values.shape, dtype=np.int64)
        special_head = np.empty(values.shape, dtype=np.float64)
        special_tail = np.empty(values.shape, dtype=np.float64)
        reduced = values - _PIO2_1
        primary = reduced - _PIO2_1T
        _store_special(
            ~neg & ~near,
            special_cycles,
            special_head,
            special_tail,
            1,
            primary,
            (reduced - primary) - _PIO2_1T,
        )
        reduced = values + _PIO2_1
        primary = reduced + _PIO2_1T
        _store_special(
            neg & ~near,
            special_cycles,
            special_head,
            special_tail,
            -1,
            primary,
            (reduced - primary) + _PIO2_1T,
        )
        reduced = (values - _PIO2_1) - _PIO2_2
        primary = reduced - _PIO2_2T
        _store_special(
            ~neg & near,
            special_cycles,
            special_head,
            special_tail,
            1,
            primary,
            (reduced - primary) - _PIO2_2T,
        )
        reduced = (values + _PIO2_1) + _PIO2_2
        primary = reduced + _PIO2_2T
        _store_special(
            neg & near,
            special_cycles,
            special_head,
            special_tail,
            -1,
            primary,
            (reduced - primary) + _PIO2_2T,
        )
        cycles[special] = special_cycles
        head[special] = special_head
        tail[special] = special_tail
        use_tail[special] = True

    medium = (absolute >= np.uint32(0x4002D97C)) & (absolute <= np.uint32(0x413921FB))
    if np.any(medium):
        values = flat[medium]
        magnitude = np.abs(values)
        medium_cycles = np.trunc(magnitude * _INVPIO2 + _HALF).astype(np.int64)
        scale = medium_cycles.astype(np.float64)
        remainder = magnitude - scale * _PIO2_1
        reduced_tail = scale * _PIO2_1T
        reduced_head = remainder - reduced_tail
        lookup = np.clip(medium_cycles - np.int64(1), 0, _NPIO2_HW_ARRAY.size - 1)
        quick = (medium_cycles >= np.int64(1)) & (medium_cycles < np.int64(32))
        quick &= absolute[medium] != _NPIO2_HW_ARRAY[lookup]
        exponent = absolute[medium].astype(np.int64) >> np.int64(20)
        second = ~quick & (exponent - _exponent_field(reduced_head) > np.int64(16))
        if np.any(second):
            saved = remainder[second]
            step = scale[second] * _PIO2_2
            second_remainder = saved - step
            second_tail = scale[second] * _PIO2_2T - ((saved - second_remainder) - step)
            second_head = second_remainder - second_tail
            third = exponent[second] - _exponent_field(second_head) > np.int64(49)
            if np.any(third):
                saved = second_remainder[third]
                step = scale[second][third] * _PIO2_3
                third_remainder = saved - step
                second_tail[third] = scale[second][third] * _PIO2_3T - ((saved - third_remainder) - step)
                second_remainder[third] = third_remainder
                second_head[third] = third_remainder - second_tail[third]
            remainder[second] = second_remainder
            reduced_tail[second] = second_tail
            reduced_head[second] = second_head
        low = (remainder - reduced_head) - reduced_tail
        neg = negative[medium]
        cycles[medium] = np.where(neg, -medium_cycles, medium_cycles)
        head[medium] = np.where(neg, -reduced_head, reduced_head)
        tail[medium] = np.where(neg, -low, low)
        use_tail[medium] = True

    shape = number.shape
    return cycles.reshape(shape), head.reshape(shape), tail.reshape(shape), use_tail.reshape(shape)


def _sincos_array(number: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cycles, head, tail, use_tail = _reduce_array(number)
    sine_tail = _kernel_sin_array(head, tail, True)
    if np.any(~use_tail):
        sine = np.where(use_tail, sine_tail, _kernel_sin_array(head, tail, False))
    else:
        sine = sine_tail
    cosine_value = _kernel_cos_array(head, tail)
    quadrant = cycles & np.int64(3)
    sine_result = np.where(
        quadrant == np.int64(0),
        sine,
        np.where(
            quadrant == np.int64(1),
            cosine_value,
            np.where(quadrant == np.int64(2), -sine_tail, -cosine_value),
        ),
    )
    cosine_result = np.where(
        quadrant == np.int64(0),
        cosine_value,
        np.where(
            quadrant == np.int64(1),
            -sine_tail,
            np.where(quadrant == np.int64(2), -cosine_value, sine_tail),
        ),
    )
    large = (_unsigned_high(number) & np.uint32(0x7FFFFFFF)) > np.uint32(0x413921FB)
    if np.any(large):
        flat = number.reshape(-1)
        sine_flat = sine_result.reshape(-1)
        cosine_flat = cosine_result.reshape(-1)
        for index in np.flatnonzero(large.reshape(-1)):
            value = flat[index]
            sine_flat[index] = _sin_scalar(value)
            cosine_flat[index] = _cos_scalar(value)
    return sine_result, cosine_result


def _sincos(values: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    requested = np.asarray(values, dtype=np.float64)
    number = np.ascontiguousarray(requested)
    with np.errstate(invalid="ignore", over="ignore"):
        sine, cosine = _sincos_array(number)
    return sine.reshape(requested.shape), cosine.reshape(requested.shape)


def sin(values: np.ndarray | float) -> np.ndarray:
    """Return fdlibm ``sin`` of each value."""
    sine, _cosine = _sincos(values)
    return sine


def cos(values: np.ndarray | float) -> np.ndarray:
    """Return fdlibm ``cos`` of each value."""
    _sine, cosine = _sincos(values)
    return cosine


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
