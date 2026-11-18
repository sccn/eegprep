"""fdlibm exp, bit for bit with the scalar reference."""

from __future__ import annotations

import numpy as np

from eegprep.functions.miscfunc.ieee754 import _O_THRESHOLD, _U_THRESHOLD, _exp_scalar, cos, exp, sin


def _bits(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64).view(np.uint64)


def _high_word(high: int, low: int = 0) -> np.float64:
    return np.uint64((np.uint64(high) << np.uint64(32)) | np.uint64(low)).view(np.float64)


def _sample() -> np.ndarray:
    boundary = np.float64.fromhex("0x1p-28")
    edges = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        np.inf,
        -np.inf,
        np.nan,
        np.nextafter(0.0, 1.0),
        np.nextafter(0.0, -1.0),
        np.finfo(np.float64).tiny,
        -np.finfo(np.float64).tiny,
        boundary,
        np.nextafter(boundary, 0.0),
        np.nextafter(boundary, 1.0),
        -boundary,
        _O_THRESHOLD,
        np.nextafter(_O_THRESHOLD, 0.0),
        np.nextafter(_O_THRESHOLD, np.inf),
        _U_THRESHOLD,
        np.nextafter(_U_THRESHOLD, 0.0),
        np.nextafter(_U_THRESHOLD, -np.inf),
    ]
    for high in (0x3FD62E42, 0x3FD62E43, 0x3FF0A2B1, 0x3FF0A2B2, 0x3E2FFFFF, 0x3E300000, 0x40862E41, 0x40862E42):
        word = _high_word(high)
        edges.extend((word, -word, _high_word(high, 0xFFFFFFFF), -_high_word(high, 0xFFFFFFFF)))
    generator = np.random.default_rng(20261118)
    uniform = generator.uniform(-745.0, 710.0, 4096)
    normal = generator.normal(scale=20.0, size=2048)
    subnormal = np.ldexp(generator.integers(1, 2**52, size=256), -1074)
    powers = np.arange(-1074, 1025) * np.log(2.0)
    return np.concatenate([np.asarray(edges, dtype=np.float64), uniform, normal, subnormal, -subnormal, powers])


def test_vector_exp_matches_the_scalar_reference():
    values = _sample()
    got = exp(values)
    expected = np.array([_exp_scalar(value) for value in values], dtype=np.float64)
    np.testing.assert_array_equal(_bits(got), _bits(expected))

    square = values[:100].reshape(10, 10)
    got_square = exp(square)
    assert got_square.shape == square.shape
    expected_square = np.array([_exp_scalar(value) for value in square.ravel()], dtype=np.float64).reshape(square.shape)
    np.testing.assert_array_equal(_bits(got_square), _bits(expected_square))

    scalar = exp(np.float64(1.0))
    assert scalar.shape == ()
    assert _bits(scalar) == _bits(_exp_scalar(1.0))
    assert exp(np.array([], dtype=np.float64)).shape == (0,)

    near_zero = np.concatenate(
        [
            np.linspace(-0.2, -1.0e-20, 2048),
            np.ldexp(np.arange(1, 129), -1074),
            np.array([0.0, -0.0]),
        ]
    )
    got_near = exp(near_zero)
    expected_near = np.array([_exp_scalar(value) for value in near_zero], dtype=np.float64)
    np.testing.assert_array_equal(_bits(got_near), _bits(expected_near))


def test_sin_and_cos_match_fdlibm_bits():
    assert sin(np.float64(-0.0)).view(np.uint64) == np.uint64(1 << 63)
    assert cos(0.0).view(np.uint64) == np.float64(1.0).view(np.uint64)
    angle = np.float64.fromhex("-0x1.acee9f37bebd6p-1")
    pair = np.array([np.float64(0.1), angle])
    assert cos(pair)[0].hex() == "0x1.fd712f9a817c0p-1"
    assert sin(pair)[1].hex() == "-0x1.7c7d7a833bec2p-1"
    assert float(cos(np.float64.fromhex("-0x1.47559eae3f79dp+26"))).hex() == "-0x1.af4718c2d6a84p-7"
    assert np.isnan(sin(np.inf))
    assert np.isnan(cos(np.nan))
    assert sin(np.array([], dtype=np.float64)).shape == (0,)
