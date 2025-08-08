"""
Bruin Livery Manager Package

A Python packge for a lot of things

Modules:
- step_convert: STEP to USD conversion functionality
  - config: Configuration settings and constants
  - name_utils: Name processing and sanitization utilities  
  - step_reader: STEP file reading and hierarchy parsing
  - geometry_processor: Geometry processing and mesh generation
  - usd_converter: USD format conversion functionality
  - utils: Utility functions for reporting and analysis
  - step_to_usd: Main entry point for STEP to USD conversion
- project_structure: Project scaffolding and management
  - config: Project structure configuration
  - project_structure: Project structure management utilities
"""

from .step_convert.config import *
from .project_structure.config import *

__version__ = "1.0.0"
__author__ = "Bruin Livery Manager"

__all__ = [
    'MESH_LINEAR_DEFLECTION', 
    'MESH_ANGULAR_DEFLECTION',
    'FLIP_NORMALS',
    'FORCE_CONSISTENT_WINDING',
    'DEFAULT_PROJECT_STRUCTURE_YML'
]