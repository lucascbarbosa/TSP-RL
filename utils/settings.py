"""Settings and environment variable management for TSP-RL project."""
import os
from typing import Optional


def get_gurobi_credentials() -> dict:
    """
    Get Gurobi credentials from environment variables.
    
    Returns:
        dict: Dictionary with WLSACCESSID, WLSSECRET, and LICENSEID
    """
    wls_access_id = os.getenv('WLSACCESSID')
    wls_secret = os.getenv('WLSSECRET')
    license_id = os.getenv('LICENSEID')
    
    if not all([wls_access_id, wls_secret, license_id]):
        raise ValueError(
            "Gurobi credentials not found in environment variables. "
            "Please set WLSACCESSID, WLSSECRET, and LICENSEID."
        )
    
    params = {
        "WLSACCESSID": wls_access_id,
        "WLSSECRET": wls_secret,
        "LICENSEID": int(license_id)
    }
    
    return params


def get_tsp_data_path() -> Optional[str]:
    """
    Get path to TSP data directory from environment variable.
    
    Returns:
        Optional[str]: Path to TSP data directory, or None if not set
    """
    return os.getenv('TSP_DATA_PATH')


# Default configuration values
DEFAULT_TIME_LIMIT = 60
DEFAULT_MIP_GAP = 0.01
DEFAULT_THREADS = 4
DEFAULT_VERBOSE = True

