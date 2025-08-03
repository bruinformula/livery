# Livery

A comprehensive control ship scheme for livery projects, providing automated tools and workflows for converting CAD assets into production-ready USD scenes. Currently focused on STEP to USD conversion with hierarchy preservation, but designed to expand into a full pipeline for material assignment, subdivision surface preparation, proxy generation, and project management.

## Current Features

- **STEP to USD Conversion**: Converts STEP files from SolidWorks and other CAD software to USD format
- **Hierarchical Structure Preservation**: Maintains the assembly hierarchy from STEP files in the USD output
- **Reference Duplication Handling**: Intelligently processes repeated components and references
- **Mesh Quality Control**: Configurable tessellation parameters for optimal quality/performance balance
- **Metadata Preservation**: Extracts and preserves component names and properties from STEP files
- **Robust Error Handling**: Graceful fallbacks for complex or malformed STEP files

## Planned Features (Roadmap)

- **Automated Material Assignment**: Intelligent material mapping based on component names and metadata
- **Material Libraries**: Curated collections of relevant materials
- **Edge Assignment for Subdivision**: Automatic detection and tagging of edges for subdivision surface workflows
- **Proxy Mesh Generation**: Multi-resolution LOD generation for performance optimization
- **Project Directory Creation**: Standardized project structure setup with proper USD stage organization
## Requirements

This project requires OpenUSD (built from source), OpenCascade, and their Python bindings. You can use either standard Python virtual environments or conda (recommended for OpenCascade due to easier build).

### Verification

Test your installation:

```python
# Test USD
from pxr import Usd, UsdGeom
print("USD installed successfully")

# Test OpenCascade bindings
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Extend.DataExchange import read_step_file
print("OpenCascade bindings working")
```

## Usage

### Basic Conversion

Convert a STEP file to USD format:

```bash
python convert_step_to_USD.py
```

By default, this processes `./test_inputs/MK10.STEP` and outputs to `./output/output.usda`.

### Custom File Paths

Modify the file paths in `convert_step_to_USD.py`:

```python
step_file = "path/to/your/model.STEP"
usd_file = "path/to/output/model.usda"
main(step_file, usd_file)
```

### Configuration

Adjust mesh quality and conversion settings in `step_convert/config.py`:

```python
# Mesh tessellation quality
MESH_LINEAR_DEFLECTION = 1.0    # Lower = higher quality, more triangles
MESH_ANGULAR_DEFLECTION = 0.5   # Lower = smoother curves

# Normal orientation
FLIP_NORMALS = False             # Set True if faces appear inside-out
FORCE_CONSISTENT_WINDING = True  # Ensure consistent triangle orientation
```

## Project Structure

```
livery/
├── convert_step_to_USD.py          # Main conversion script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── step_convert/                   # Core conversion modules
│   ├── __init__.py
│   ├── calculate_normals.py        # Normal vector calculations
│   ├── config.py                   # Configuration settings
│   ├── geometry_processor.py       # Mesh processing utilities
│   ├── name_utils.py               # STEP metadata extraction
│   ├── step_reader.py              # STEP file parsing with hierarchy
│   ├── usd_converter.py            # USD generation logic
│   └── utils.py                    # Utility functions
├── test_inputs/                    # Sample STEP files
│   └── test.STEP
└── output/                         # Generated USD files
```

## How It Works

### Current Implementation (STEP to USD)

1. **STEP File Analysis**: Extracts product names and hierarchy information from STEP files
2. **Hierarchy Reconstruction**: Builds a tree structure representing the assembly hierarchy
3. **Geometry Processing**: Tessellates CAD surfaces into meshes with configurable quality
4. **USD Generation**: Creates USD prims with proper transforms and mesh data
5. **Metadata Preservation**: Maintains component names and relationships from the source

The converter handles complex assemblies with nested components and automatically detects and processes reference duplications common in CAD assemblies.

### Future Pipeline Vision

The livery control ship will expand to include:
- **Intelligent Material Assignment**: Using component naming conventions and metadata to automatically assign appropriate materials
- **Subdivision Surface Preparation**: Analyzing geometry to identify and tag edges suitable for subdivision workflows
- **Multi-Resolution Asset Generation**: Creating proxy meshes and LOD variants for different use cases
- **Project Standardization**: Enforcing consistent USD stage organization and naming conventions