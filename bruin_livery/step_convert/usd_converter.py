"""USD conversion and export functionality.

TRANSFORMATION HANDLING:
This module now handles transformations in USD space rather than baking them into geometry.

Key changes:
1. Geometry is kept in local coordinate space (no apply_location_to_shape)
2. Each USD Xform prim gets the transformation from its STEP component
3. Transformations are preserved as USD hierarchy rather than baked into vertices
4. This enables proper manipulation, animation, and instancing in USD-aware applications

Benefits:
- Preserves parametric transformation hierarchy
- Enables USD animation and manipulation
- Supports proper instancing and referencing
- More faithful to original STEP assembly structure
"""

from __future__ import annotations
from typing import Any, MutableSet, Optional, List, Dict, Tuple, Union, Protocol

from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.XCAFDoc import XCAFDoc_ShapeTool
from OCC.Core.gp import gp_Trsf, gp_Pnt
from OCC.Core.Poly import Poly_Triangulation, Poly_Triangle
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID
from pxr import Usd, UsdGeom, Gf, Sdf, Vt

from .name_utils import sanitize_usd_name, generate_unique_name
from .geometry_processor import triangulate_shape, extract_faces, apply_location_to_shape
from .calculate_normals import calculate_face_normals, calculate_parametic_normals, ensure_consistent_winding_order
from .step_reader import STEPFile, ComponentInfo

# Type aliases for better readability
Point3D = Tuple[float, float, float]
Normal3D = Tuple[float, float, float]
FaceData = Tuple[TopoDS_Face, Optional[Poly_Triangulation]]
VertexMap = Dict[Point3D, int]

# Protocol for shape tool to provide better typing
class ShapeToolProtocol(Protocol):
    """Protocol for XCAFDoc_ShapeTool to provide better typing without circular imports."""
    pass

try:
    from .config import FORCE_CONSISTENT_WINDING
except ImportError:
    # Default values if config not available
    FLIP_NORMALS: bool = False
    FORCE_CONSISTENT_WINDING: bool = True

def convert_hierarchical_shape_to_usd(
    stage: Usd.Stage, 
    parent_prim: UsdGeom.Xform, 
    shape_info: ComponentInfo, 
    shape_tool: XCAFDoc_ShapeTool,
    depth: int = 0, 
    accumulated_transform: Optional[TopLoc_Location] = None
) -> UsdGeom.Xform:
    """Convert a hierarchical shape to USD, with proper transformation accumulation.
    
        Args:
        stage: USD stage
        parent_prim: Parent USD prim 
        shape_info: ComponentInfo containing shape data and hierarchy
        shape_tool: XCAF shape tool for additional operations
        depth: Current depth in hierarchy for indentation
        accumulated_transform: Accumulated transformation from parent hierarchy
        
    Returns:
        The created USD Xform prim
    """
    
    indent: str = "  " * depth
    nauo_info: str = f" (NAUO: {shape_info.nauo_id})" if shape_info.nauo_id else ""
    product_info: str = f" (PRODUCT: {shape_info.product_name})" if shape_info.product_name else ""
    
    usd_object_name: str
    if shape_info.product_name:
        usd_object_name = shape_info.product_name
    else:
        usd_object_name = shape_info.name
    
    unique_name: str
    if shape_info.nauo_id:
        unique_name = f"{sanitize_usd_name(usd_object_name)}_{shape_info.nauo_id}"
    else:
        unique_name = sanitize_usd_name(usd_object_name)
        if not unique_name or unique_name == "unnamed":
            unique_name = generate_unique_name()
    
    sanitized_name: str = unique_name
    
    xform_path: Sdf.Path = parent_prim.GetPath().AppendChild(sanitized_name)
    xform: UsdGeom.Xform = UsdGeom.Xform.Define(stage, xform_path)
    
    current_location: TopLoc_Location = shape_info.location
    if accumulated_transform is None:
        accumulated_transform: TopLoc_Location = TopLoc_Location()
    
    if current_location and not current_location.IsIdentity():
        total_transform: TopLoc_Location = accumulated_transform * current_location
    else:
        total_transform: TopLoc_Location = accumulated_transform
    
    prim: Usd.Prim = xform.GetPrim()
    if shape_info.nauo_id:
        prim.SetCustomDataByKey('nauo_id', shape_info.nauo_id)
    if shape_info.product_name:
        prim.SetCustomDataByKey('product_name', shape_info.product_name)
    
    prim.SetDisplayName(shape_info.name)
    
    if not shape_info.children and not shape_info.is_assembly:
        shape: TopoDS_Shape = shape_info.shape
        
        if total_transform and not total_transform.IsIdentity():
            shape: TopoDS_Shape = apply_location_to_shape(shape, total_transform)
        
        explorer: TopExp_Explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        if explorer.More():
            mesh_base_name: str = shape_info.product_name or shape_info.name
            mesh_name: str = f"{sanitize_usd_name(mesh_base_name)}_mesh"
            try:
                mesh_prim: Optional[UsdGeom.Mesh] = convert_shape_to_usd_mesh(stage, xform, shape, mesh_name)
                if mesh_prim:
                    # Get mesh statistics for display
                    mesh_points: Optional[Any] = mesh_prim.GetPointsAttr().Get()
                    mesh_face_counts: Optional[Any] = mesh_prim.GetFaceVertexCountsAttr().Get()
                    vertex_count: int = len(mesh_points) if mesh_points else 0
                    face_count: int = len(mesh_face_counts) if mesh_face_counts else 0
                    print(f"{indent}Created: {mesh_name} with {vertex_count} vertices and {face_count} faces")
            except Exception as e:
                print(f"{indent}Warning: Failed to create mesh for {shape_info.name}: {e}")
        else:
            print(f"{indent}Part {shape_info.name} has no solid geometry (might be surface/wire)")
    else:
        print(f"{indent}Processing: {shape_info.name}{nauo_info}{product_info}")
    
    if shape_info.children:
        for i, child_info in enumerate(shape_info.children):
            try:
                convert_hierarchical_shape_to_usd(stage, xform, child_info, shape_tool, depth + 1, total_transform)
            except Exception as e:
                print(f"{indent}  Warning: Failed to convert child {i} of {shape_info.name}: {e}")
    
    return xform

def convert_shape_to_usd_mesh(
    stage: Usd.Stage, 
    parent_prim: UsdGeom.Xform, 
    shape: TopoDS_Shape, 
    name: str
) -> Optional[UsdGeom.Mesh]:
    """Convert a triangulated shape to a USD Mesh.
    
    Args:
        stage: USD stage
        parent_prim: Parent USD prim
        shape: OpenCASCADE shape to convert
        name: Name for the mesh
        
    Returns:
        Created USD Mesh or None if conversion failed
    """
    
    triangulate_shape(shape)

    faces: List[FaceData] = extract_faces(shape)
    
    if not faces:
        print(f"    Warning: No triangulated faces found for shape {name}")
        return None
    
    all_points: List[Point3D] = []
    all_face_vertex_counts: List[int] = []
    all_face_vertex_indices: List[int] = []
    all_normals: List[Normal3D] = []
    vertex_map: VertexMap = {}
    next_vertex_index: int = 0
    
    for face, triangulation in faces:
        if triangulation is None:
            continue
                

        face_normals: List[float] = calculate_face_normals(face, triangulation, None)
        face_normals: List[float] = calculate_parametic_normals(face, triangulation, None)

        face_vertex_map: Dict[int, int] = {}
        
        for i in range(1, triangulation.NbNodes() + 1):
            node: gp_Pnt = triangulation.Node(i)
            vertex: Point3D = (float(node.X()), float(node.Y()), float(node.Z()))
            
            vertex_key: Point3D = tuple(round(coord, 6) for coord in vertex)
            
            if vertex_key not in vertex_map:
                vertex_map[vertex_key] = next_vertex_index
                all_points.append(vertex)
                next_vertex_index += 1
            
            face_vertex_map[i] = vertex_map[vertex_key]
        
        normal_index: int = 0
        for i in range(1, triangulation.NbTriangles() + 1):
            triangle: Poly_Triangle = triangulation.Triangle(i)
            n1, n2, n3 = triangle.Get()
            
            if n1 in face_vertex_map and n2 in face_vertex_map and n3 in face_vertex_map:
                idx1, idx2, idx3 = face_vertex_map[n1], face_vertex_map[n2], face_vertex_map[n3]
                if idx1 != idx2 and idx2 != idx3 and idx1 != idx3:
                    all_face_vertex_counts.append(3)
                    all_face_vertex_indices.extend([idx1, idx2, idx3])
                    
                    if normal_index * 3 + 2 < len(face_normals):
                        all_normals.extend([
                            face_normals[normal_index * 3],
                            face_normals[normal_index * 3 + 1], 
                            face_normals[normal_index * 3 + 2]
                        ])
                    else:
                        default_normal: Normal3D = (0.0, 0.0, 1.0)
                        all_normals.extend([default_normal, default_normal, default_normal])
                    
                    normal_index += 1
    
    if not all_points:
        print(f"    Warning: No vertices found for shape {name}")
        return None

    if FORCE_CONSISTENT_WINDING and all_normals:
        all_face_vertex_indices, all_normals = ensure_consistent_winding_order(
            all_points, all_face_vertex_indices, all_normals
        )
    
    mesh_path: Sdf.Path = parent_prim.GetPath().AppendChild(name)
    mesh: UsdGeom.Mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    
    # Set basic mesh attributes
    mesh.CreatePointsAttr().Set(all_points)
    mesh.CreateFaceVertexCountsAttr().Set(all_face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr().Set(all_face_vertex_indices)
    
    if all_normals:
        mesh.CreateNormalsAttr().Set(all_normals)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    
    return mesh



def gp_trsf_to_usd_matrix(trsf: gp_Trsf) -> Gf.Matrix4d:
    """Convert gp_Trsf to USD 4x4 matrix with proper precision.
    
    Args:
        trsf: OpenCASCADE transformation
        
    Returns:
        USD 4x4 transformation matrix
    """
    # Get the transformation matrix components (OpenCASCADE uses 1-based indexing)
    # USD expects row-major 4x4 matrix
    translation = trsf.TranslationPart()
    
    # Build the 4x4 transformation matrix with proper float precision
    # Row 0: [m11, m12, m13, tx]
    # Row 1: [m21, m22, m23, ty] 
    # Row 2: [m31, m32, m33, tz]
    # Row 3: [0,   0,   0,   1]
    
    # Clean up very small values to avoid -0.0 display issues
    def clean_float(val: float) -> float:
        return 0.0 if abs(val) < 1e-10 else float(val)
    
    return Gf.Matrix4d(
        clean_float(trsf.Value(1, 1)), clean_float(trsf.Value(1, 2)), clean_float(trsf.Value(1, 3)), clean_float(translation.X()),
        clean_float(trsf.Value(2, 1)), clean_float(trsf.Value(2, 2)), clean_float(trsf.Value(2, 3)), clean_float(translation.Y()),
        clean_float(trsf.Value(3, 1)), clean_float(trsf.Value(3, 2)), clean_float(trsf.Value(3, 3)), clean_float(translation.Z()),
        0.0, 0.0, 0.0, 1.0
    )

def toploc_to_usd_matrix(location: TopLoc_Location) -> Gf.Matrix4d:
    """Convert TopLoc_Location to USD 4x4 matrix.
    
    Args:
        location: OpenCASCADE location/transformation
        
    Returns:
        USD 4x4 transformation matrix
    """
    if location.IsIdentity():
        return Gf.Matrix4d(1.0)  # Identity matrix
    
    trsf: gp_Trsf = location.Transformation()
    return gp_trsf_to_usd_matrix(trsf)

