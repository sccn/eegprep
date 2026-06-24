import threadpoolctl

def get_math_backend_info():
    try:
        import numpy  # noqa: F401
    except ImportError:
        pass

    info = threadpoolctl.threadpool_info()
    return info

def check_conflicting_libraries(info):
    """
    Returns a list of warning dictionaries if multiple conflicting math libraries 
    (e.g., multiple different internal APIs for BLAS) are loaded.
    """
    blas_apis = set()
    lapack_apis = set()
    
    for item in info:
        user_api = item.get("user_api")
        internal_api = item.get("internal_api")
        if user_api == "blas" and internal_api:
            blas_apis.add(internal_api)
        if user_api == "lapack" and internal_api:
            lapack_apis.add(internal_api)
            
    warnings = []
    if len(blas_apis) > 1:
        warnings.append({
            "code": "CONFLICTING_BLAS_LIBRARIES",
            "message": f"Multiple conflicting BLAS libraries detected: {', '.join(sorted(blas_apis))}",
            "severity": "warning",
            "suggestion": "Ensure only one BLAS implementation is loaded to avoid numerical drift."
        })
    if len(lapack_apis) > 1:
        warnings.append({
            "code": "CONFLICTING_LAPACK_LIBRARIES",
            "message": f"Multiple conflicting LAPACK libraries detected: {', '.join(sorted(lapack_apis))}",
            "severity": "warning",
            "suggestion": "Ensure only one LAPACK implementation is loaded to avoid numerical drift."
        })
        
    return warnings
