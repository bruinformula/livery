"""Configuration settings for STEP to USD conversion."""

# Mesh quality settings - balanced for good quality without over-triangulation
MESH_LINEAR_DEFLECTION = 1.75   # Reasonable tessellation for most CAD models  
MESH_ANGULAR_DEFLECTION = 0.25  # Good angular resolution without excessive triangles

# Alternative settings for different quality needs:
# High quality (slower, more triangles):
# MESH_LINEAR_DEFLECTION = 0.1
# MESH_ANGULAR_DEFLECTION = 0.1

# Lower quality (faster, fewer triangles):
# MESH_LINEAR_DEFLECTION = 2.0
# MESH_ANGULAR_DEFLECTION = 1.0

# Normal orientation settings
FLIP_NORMALS = False  # Set to True if most faces appear inverted (inside-out)
FORCE_CONSISTENT_WINDING = True  # Ensure consistent triangle winding order
ENABLE_USD_UV = True 
ENABLE_USD_CREASES = False  # Enable USD crease attributes for sharp edges

# Sharp edge detection settings
ENABLE_SHARP_EDGE_DETECTION = True  # Enable analysis of surface derivatives for sharp edges
USE_FACE_VARYING_NORMALS = False  # Use face-varying normals for sharp edges (creates harder edges)

# UV coordinate settings
ENABLE_UV_COORDINATES = True  # Enable UV coordinate generation from parametric surfaces
ENABLE_UV_SEAMS = True  # Handle UV seams at sharp edges and surface boundaries
UV_SCALE_FACTOR = 0.1  # Global scaling factor for UV coordinates bigger results in less repeatition
