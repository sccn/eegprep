"""Medium-scale BFGS ``fminunc`` used by dipole fitting and electrode warp.

FieldTrip calls MATLAB ``fminunc`` with ``LargeScale`` off and ``HessUpdate``
``bfgs``. The forward difference is ``sqrt(eps)*sign'(x)*max(|x|, TypicalX)``
with ``sign'(0) = 1``. Iteration stops when
``||g||_inf < TolFun*(1 + ||g0||_inf)`` or
``||dx / (1 + |x|)||_inf < TolX``.
"""

import numpy as np


_EPS = np.finfo(float).eps
_ARMIJO = 0.01
_WOLFE_CURVATURE = 0.9
_LINE_SEARCH_ITERATIONS = 40


class _EvaluationLimit(Exception):
    pass


def fminunc(fun, start, *, maxiter, tol_fun, tol_x, typical_x, max_fun_evals, return_status=False):
    """Minimize ``fun`` with the medium-scale BFGS update FieldTrip requests.

    Args:
        fun: Scalar objective of a one-dimensional parameter vector.
        start: Initial parameters.
        maxiter: MATLAB ``MaxIter``.
        tol_fun: MATLAB ``TolFun``.
        tol_x: MATLAB ``TolX``.
        typical_x: MATLAB ``TypicalX``. A scalar is applied to every parameter.
        max_fun_evals: MATLAB ``MaxFunEvals``, counting finite-difference probes.
        return_status: Also return the stop report. ``converged`` is true when
            the stop is the ``TolFun`` gradient test or the ``TolX`` step test.

    Returns:
        The last accepted parameter vector. An evaluation budget hit during a
        trial keeps that last accepted point. With ``return_status``, a
        ``(parameters, status)`` pair.
    """

    def counted(values):
        nonlocal evaluations
        evaluations += 1
        if evaluations > max_fun_evals:
            evaluations -= 1
            raise _EvaluationLimit
        return float(fun(np.asarray(values, dtype=float)))

    def grad(point, known):
        step = np.sqrt(_EPS) * np.where(point < 0, -1.0, 1.0) * np.maximum(np.abs(point), typical_x)
        return np.array(
            [(counted(point + np.eye(len(point))[i] * step[i]) - known) / step[i] for i in range(len(point))]
        )

    values = np.array(start, dtype=float).ravel()
    evaluations = 0
    reason = "maxiter"
    try:
        value = counted(values)
        gradient = grad(values, value)
        initial_gradient = np.linalg.norm(gradient, np.inf)
        inverse = np.eye(len(values))
        for _iteration in range(maxiter):
            if np.linalg.norm(gradient, np.inf) < tol_fun * (1 + initial_gradient):
                reason = "gradient"
                break
            direction = -inverse @ gradient
            first_step = min(1 / initial_gradient, 1.0) if _iteration == 0 and initial_gradient > 0 else 1.0
            alpha, new_value, new_gradient = _line_search(counted, grad, values, direction, gradient, value, first_step)
            if new_value >= value:
                reason = "no_decrease"
                break
            delta = alpha * direction
            values = values + delta
            delta_gradient = new_gradient - gradient
            value, gradient = new_value, new_gradient
            if np.linalg.norm(delta / (1 + np.abs(values)), np.inf) < tol_x:
                reason = "step"
                break
            curvature = delta @ delta_gradient
            if curvature >= np.sqrt(_EPS) * max(_EPS, np.linalg.norm(delta) * np.linalg.norm(delta_gradient)):
                if _iteration == 0:
                    inverse = curvature / (delta_gradient @ delta_gradient) * np.eye(len(values))
                projected = inverse @ delta_gradient
                inverse += (1 + delta_gradient @ projected / curvature) * np.outer(delta, delta) / curvature - (
                    np.outer(delta, projected) + np.outer(projected, delta)
                ) / curvature
    except _EvaluationLimit:
        reason = "max_fun_evals"
    if not return_status:
        return values
    return values, {"converged": reason in {"gradient", "step"}, "reason": reason, "evaluations": evaluations}


def _cubic(a, fa, ga, b, fb, gb):
    if a == b:
        return a
    slope = (fa - fb) / (a - b)
    z = ga + gb - 3 * slope
    rad = z * z - ga * gb
    if rad < 0:
        return (a + b) / 2
    w = np.sign(b - a) * np.sqrt(rad)
    denominator = gb - ga + 2 * w
    if denominator == 0:
        return (a + b) / 2
    return b - (b - a) * (gb + w - z) / denominator


def _section_trial(trial, anchor, other):
    """Clip a sectioning trial to 10% from ``anchor`` and 50% from ``other``."""
    low, high = sorted((anchor + 0.1 * (other - anchor), other - 0.5 * (other - anchor)))
    return float(min(max(trial, low), high))


def _line_search(fun, grad, x, p, g, f, first_step=1.0):
    """Strong Wolfe search. ``a`` is the last step that met sufficient decrease."""
    derivative = g @ p
    a, fa, ga = 0.0, f, derivative
    b = first_step
    last = (0.0, f, g)
    positive_slope = False
    for i in range(_LINE_SEARCH_ITERATIONS):
        fb = fun(x + b * p)
        gbvec = grad(x + b * p, fb)
        gb = gbvec @ p
        last = (b, fb, gbvec)
        if fb > f + _ARMIJO * b * derivative or (i and fb >= fa):
            break
        if abs(gb) <= -_WOLFE_CURVATURE * derivative:
            return b, fb, gbvec
        if gb >= 0:
            positive_slope = True
            break
        # MATLAB extrapolates between one and nine previous interval lengths.
        new = _cubic(a, fa, ga, b, fb, gb)
        new = np.clip(new, b + (b - a), b + 9 * (b - a)) if new > b else b + 9 * (b - a)
        a, fa, ga = b, fb, gb
        b = new
    if positive_slope:
        a, fa, ga, b, fb, gb = b, fb, gb, a, fa, ga
    roundoff = np.spacing(max(1.0, abs(f)))
    for _ in range(_LINE_SEARCH_ITERATIONS):
        c = _section_trial(_cubic(a, fa, ga, b, fb, gb), a, b)
        if (a - c) * ga <= roundoff:
            return c, last[1], last[2]
        fc = fun(x + c * p)
        gcvec = grad(x + c * p, fc)
        gc = gcvec @ p
        last = (c, fc, gcvec)
        if fc > f + _ARMIJO * c * derivative or fc >= fa:
            b, fb, gb = c, fc, gc
        else:
            if abs(gc) <= -_WOLFE_CURVATURE * derivative:
                return c, fc, gcvec
            if gc * (b - a) >= 0:
                b, fb, gb = a, fa, ga
            a, fa, ga = c, fc, gc
    return last
