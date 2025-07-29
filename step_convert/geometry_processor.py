"""Geometry processing and mesh generation functionality."""

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

try:
    from .config import MESH_LINEAR_DEFLECTION, MESH_ANGULAR_DEFLECTION
except ImportError:
    from config import MESH_LINEAR_DEFLECTION, MESH_ANGULAR_DEFLECTION


def extract_solids_from_shape(shape):
    """Extract individual solid bodies from a compound shape."""
    solids = []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        solid = explorer.Current()
        solids.append(solid)
        explorer.Next()
    
    # If no solids found, return the original shape
    if not solids:
        return [shape]
    
    return solids


def triangulate_shape(shape):
    """Triangulate a shape for export."""
    mesh = BRepMesh_IncrementalMesh(shape, MESH_LINEAR_DEFLECTION, False, MESH_ANGULAR_DEFLECTION, True)
    mesh.Perform()


def extract_faces(shape):
    """Extract mesh data from triangulated shape."""
    faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is not None:
            faces.append((face, tri))
        explorer.Next()
    return faces


def apply_location_to_shape(shape, location):
    """Apply a TopLoc_Location to a shape."""
    if location.IsIdentity():
        return shape
    
    trsf = location.Transformation()
    transform_op = BRepBuilderAPI_Transform(shape, trsf, True)
    return transform_op.Shape()
