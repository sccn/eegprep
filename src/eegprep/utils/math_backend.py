import threadpoolctl


def get_math_backend_info():
    try:
        import numpy  # noqa: F401
    except ImportError:
        pass

    info = threadpoolctl.threadpool_info()
    return info
