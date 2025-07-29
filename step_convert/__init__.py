"""
STEP to USD Converter Package

A modular Python package for converting STEP files to USD format while
preserving hierarchical structure and metadata.

Modules:
- config: Configuration settings and constants
- name_utils: Name processing and sanitization utilities  
- step_reader: STEP file reading and hierarchy parsing
- geometry_processor: Geometry processing and mesh generation
- usd_converter: USD format conversion functionality
- utils: Utility functions for reporting and analysis
"""

from .config import *

__version__ = "1.0.0"
__author__ = "STEP to USD Converter"

__all__ = [
    'MESH_LINEAR_DEFLECTION', 
    'MESH_ANGULAR_DEFLECTION',
]
