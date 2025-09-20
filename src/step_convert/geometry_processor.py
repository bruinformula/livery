"""Geometry processing and mesh generation functionality."""

from __future__ import annotations
from typing import List, Tuple, Optional

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face
from OCC.Core.Poly import Poly_Triangulation
from OCC.Core.gp import gp_Trsf

import gmsh

try:
    from .config import MESH_LINEAR_DEFLECTION, MESH_ANGULAR_DEFLECTION
except ImportError:
    from config import MESH_LINEAR_DEFLECTION, MESH_ANGULAR_DEFLECTION

FaceData = Tuple[TopoDS_Face, Poly_Triangulation]

def triangulate_shape(shape: TopoDS_Shape) -> None:
    """Triangulate a shape for export using gmsh."""
    
    #if not gmsh.isInitialized():
    #    gmsh.initialize()

    #shape_ptr = int(shape.this)
    #dimTags = gmsh.model.occ.importShapesNativePointer(shape_ptr)
    #print(f"Gmsh import result: {dimTags}")
    #gmsh.model.occ.synchronize()
    
    # Fallback to OpenCASCADE triangulation

    mesh: BRepMesh_IncrementalMesh = BRepMesh_IncrementalMesh(shape, MESH_LINEAR_DEFLECTION, True, MESH_ANGULAR_DEFLECTION, False)
    mesh.Perform()


def extract_faces(shape: TopoDS_Shape) -> List[FaceData]:
    """Extract mesh data from triangulated shape."""
    faces: List[FaceData] = []
    explorer: TopExp_Explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face: TopoDS_Face = explorer.Current()
        loc: TopLoc_Location = TopLoc_Location()
        tri: Optional[Poly_Triangulation] = BRep_Tool.Triangulation(face, loc)
        if tri is not None:
            faces.append((face, tri))
        explorer.Next()
    return faces


def apply_location_to_shape(shape: TopoDS_Shape, location: TopLoc_Location) -> TopoDS_Shape:
    """Apply a TopLoc_Location to a shape."""
    if location.IsIdentity():
        return shape
    
    trsf: gp_Trsf = location.Transformation()
    transform_op: BRepBuilderAPI_Transform = BRepBuilderAPI_Transform(shape, trsf, True)
    return transform_op.Shape()

