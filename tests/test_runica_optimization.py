import numpy as np
import pytest

from eegprep.functions.sigprocfunc.runica import runica


_REFERENCE_CASES = [
    pytest.param(
        0,
        "off",
        [
            [0.006138258426678086, 0.920038988944051, -0.007614426473801105],
            [1.245639957427586, 0.01054020803403042, -0.0025770443524238138],
            [-0.00156237277009202, -0.009895611292063627, 1.424320270769803],
        ],
        [[0.0], [0.0], [0.0]],
        id="logistic-no-bias",
    ),
    pytest.param(
        0,
        "on",
        [
            [0.006133360652087168, 0.9200026096663453, -0.0076195912896603],
            [1.2455962713117488, 0.01054249400139514, -0.002576216189956501],
            [-0.0015617542160055618, -0.009908514977658586, 1.4242983431233893],
        ],
        [[0.0027986461069580363], [-0.0013498284793111981], [0.0006208222287476582]],
        id="logistic-bias",
    ),
    pytest.param(
        -1,
        "off",
        [
            [0.016542806300490903, 0.4837462583100016, -0.009894727973969961],
            [1.2466400017060382, 0.009666327764259823, 0.002293221790050116],
            [-0.004086194330539649, -0.011541930062800668, 1.2120278671034428],
        ],
        [[0.0], [0.0], [0.0]],
        id="extended-no-bias",
    ),
    pytest.param(
        -1,
        "on",
        [
            [0.01657556669489547, 0.48362814000133625, -0.009935969624553129],
            [1.24643159652647, 0.009643847125636948, 0.002320749638652659],
            [-0.0040687687430069844, -0.011580978864895756, 1.211819845044342],
        ],
        [[0.005632991536067746], [-0.008324510388924372], [0.0033542275952815784]],
        id="extended-bias",
    ),
]


@pytest.mark.parametrize(("extended", "bias", "expected_weights", "expected_bias"), _REFERENCE_CASES)
def test_optimized_training_modes_match_preoptimization_reference(extended, bias, expected_weights, expected_bias):
    """Lock down all four optimized training branches against the old implementation."""
    data = np.random.RandomState(20260716).standard_normal((3, 240)) * np.array([[1.0], [2.0], [0.5]])

    weights, sphere, _compvars, actual_bias, signs, lrates = runica(
        data,
        extended=extended,
        bias=bias,
        sphering="none",
        block=24,
        maxsteps=3,
        verbose=False,
        rndreset="off",
    )

    np.testing.assert_allclose(weights, expected_weights, rtol=5e-13, atol=5e-13)
    np.testing.assert_allclose(actual_bias, expected_bias, rtol=5e-13, atol=5e-13)
    np.testing.assert_array_equal(sphere, np.eye(3))
    np.testing.assert_array_equal(signs, [1.0, -1.0, 1.0])
    np.testing.assert_allclose(lrates, np.full(3, 0.0005916554973074442), rtol=0.0, atol=1e-18)
