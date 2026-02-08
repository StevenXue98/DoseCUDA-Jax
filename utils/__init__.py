"""
DoseCUDA-Jax utilities package.

Available modules:
- matrad_converter: Convert matRad .mat files to DoseCUDA format
"""

from .matrad_converter import (
    MatRadData,
    MatRadTestData,
    convert_to_dosecuda,
    create_dosecuda_objects,
    list_available_data,
    get_phantom_path,
    get_testdata_path,
    MATRAD_PHANTOMS,
    MATRAD_TESTDATA,
    MATRAD_PHANTOMS_DIR,
    MATRAD_TESTDATA_DIR,
)

__all__ = [
    'MatRadData',
    'MatRadTestData',
    'convert_to_dosecuda',
    'create_dosecuda_objects',
    'list_available_data',
    'get_phantom_path',
    'get_testdata_path',
    'MATRAD_PHANTOMS',
    'MATRAD_TESTDATA',
    'MATRAD_PHANTOMS_DIR',
    'MATRAD_TESTDATA_DIR',
]
