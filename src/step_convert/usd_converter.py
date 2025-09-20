
from __future__ import annotations
from typing import Any, Optional, List, Dict, Tuple, Protocol

from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.XCAFDoc import XCAFDoc_ShapeTool
from OCC.Core.gp import gp_Trsf, gp_Pnt
from OCC.Core.Poly import Poly_Triangulation, Poly_Triangle
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID
from pxr import Usd, UsdGeom, Gf, Sdf, Tf

from .name_utils import sanitize_usd_name, generate_unique_name
from .geometry_processor import triangulate_shape, extract_faces
from .calculate_normals import calculate_parametic_normals, ensure_consistent_winding_order
from .calculate_uv import generate_face_varying_uv_coordinates, calculate_parametric_uv_coordinates
from .edge_analysis import analyze_step_edges, generate_face_varying_normals
from .step_reader import STEPFile, ComponentInfo
from .crease_analysis import generate_usd_creases_from_edges


# Type aliases for better readability
Point3D = Tuple[float, float, float]
Normal3D = Tuple[float, float, float]
FaceData = Tuple[TopoDS_Face, Optional[Poly_Triangulation]]
VertexMap = Dict[Point3D, int]

# Global registry for referencing system
_geometry_registry: Dict[str, Sdf.Path] = {}
_reference_counter: Dict[str, int] = {}

def clear_geometry_registry():
    """Clear the geometry registry for a new conversion."""
    global _geometry_registry, _reference_counter
    _geometry_registry.clear()
    _reference_counter.clear()

# Protocol for shape tool to provide better typing
class ShapeToolProtocol(Protocol):
    """Protocol for XCAFDoc_ShapeTool to provide better typing without circular imports."""
    pass

try:
    from .config import FORCE_CONSISTENT_WINDING, ENABLE_SHARP_EDGE_DETECTION, USE_FACE_VARYING_NORMALS, ENABLE_USD_CREASES, ENABLE_UV_COORDINATES
except ImportError:
    # Default values if config not available
    FLIP_NORMALS: bool = False
    FORCE_CONSISTENT_WINDING: bool = True
    ENABLE_SHARP_EDGE_DETECTION: bool = True
    USE_FACE_VARYING_NORMALS: bool = True
    ENABLE_USD_CREASES: bool = True  # NEW CONFIG OPTION
    ENABLE_UV_COORDINATES: bool = True  # NEW CONFIG OPTION


def convert_hierarchical_shape_to_usd(
    stage: Usd.Stage, 
    parent_prim: UsdGeom.Xform, 
    shape_info: ComponentInfo, 
    shape_tool: XCAFDoc_ShapeTool,
    depth: int = 0, 
    accumulated_transform: Optional[TopLoc_Location] = None
) -> UsdGeom.Xform:
    """Convert a hierarchical shape to USD, with proper transformation accumulation and USD references.
    
    Args:
        stage: USD stage
        parent_prim: Parent USD prim 
        shape_info: ComponentInfo containing shape data and hierarchy
        shape_tool: XCAF shape tool for additional operations
        depth: Current depth in hierarchy for indentation
        accumulated_transform: Accumulated transformation from parent hierarchy
        library_prim: Optional library scope for storing master geometries
        
    Returns:
        The created USD Xform prim
    """
    
    # No library layer: masters are created in-place at first occurrence
    
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
    
    # Set transformation on this xform if we have a component location
    if shape_info.location and not shape_info.location.IsIdentity():
        transform_matrix = toploc_to_usd_matrix(shape_info.location)
        xform.AddTransformOp().Set(transform_matrix)
    else:
        print(f"{indent}No transform applied for {shape_info.name} (identity or missing)")
    
    prim: Usd.Prim = xform.GetPrim()
    if shape_info.nauo_id:
        prim.SetCustomDataByKey('nauo_id', shape_info.nauo_id)
    if shape_info.product_name:
        prim.SetCustomDataByKey('product_name', shape_info.product_name)
    
    prim.SetDisplayName(shape_info.name)
    
    # Handle leaf geometry nodes (parts with no children and containing solid geometry)
    if not shape_info.children and not shape_info.is_assembly:
        # Check if this is a referenced geometry
        referred_entry = shape_info.referred_entry

        mesh_base_name: str = shape_info.product_name or shape_info.name
        mesh_name: str = f"{sanitize_usd_name(mesh_base_name)}_mesh"

        if referred_entry and referred_entry in _geometry_registry:
            # This is an instance of existing geometry - create a reference as a child prim
            master_path = _geometry_registry[referred_entry]
            _reference_counter[referred_entry] += 1

            print(f"{indent}Creating reference to {master_path} for {shape_info.name} (instance #{_reference_counter[referred_entry]})")
            print(f"{indent}  🔗 Instance will inherit master geometry with own transform")

            # Create a child prim under this xform and add a reference to the master mesh
            ref_mesh_path = xform.GetPath().AppendChild(mesh_name)
            ref_mesh_prim = UsdGeom.Mesh.Define(stage, ref_mesh_path)
            ref_mesh_prim.GetPrim().GetReferences().AddInternalReference(master_path)

        else:
            # This is the first occurrence - create the master geometry as a child of this xform
            shape: TopoDS_Shape = shape_info.shape
            explorer: TopExp_Explorer = TopExp_Explorer(shape, TopAbs_SOLID)
            if explorer.More():
                if referred_entry:
                    master_path = xform.GetPath().AppendChild(mesh_name)
                    _geometry_registry[referred_entry] = master_path
                    _reference_counter[referred_entry] = 1
                    print(f"{indent}Creating master geometry at {master_path} for {shape_info.name}")
                    print(f"{indent}  📍 Master will be created in-place at first occurrence")
                    try:
                        mesh_prim: Optional[UsdGeom.Mesh] = convert_shape_to_usd_mesh(stage, xform, shape, mesh_name)
                        if mesh_prim:
                            mesh_points: Optional[Any] = mesh_prim.GetPointsAttr().Get()
                            mesh_face_counts: Optional[Any] = mesh_prim.GetFaceVertexCountsAttr().Get()
                            vertex_count: int = len(mesh_points) if mesh_points else 0
                            face_count: int = len(mesh_face_counts) if mesh_face_counts else 0
                            print(f"{indent}Created master: {mesh_name} with {vertex_count} vertices and {face_count} faces")
                    except Exception as e:
                        print(f"{indent}Warning: Failed to create master mesh for {shape_info.name}: {e}")
                else:
                    # Single occurrence - create geometry directly
                    try:
                        mesh_prim: Optional[UsdGeom.Mesh] = convert_shape_to_usd_mesh(stage, xform, shape, mesh_name)
                        if mesh_prim:
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
    
    # Process children
    if shape_info.children:
        for i, child_info in enumerate(shape_info.children):
            if shape_info.location and not shape_info.location.IsIdentity():
                transform_matrix = toploc_to_usd_matrix(shape_info.location)
            try:
                convert_hierarchical_shape_to_usd(stage, xform, child_info, shape_tool, depth + 1, accumulated_transform)
            except Exception as e:
                print(f"{indent}  Warning: Failed to convert child {i} of {shape_info.name}: {e}")
    
    return xform

def convert_shape_to_usd_mesh(
    stage: Usd.Stage, 
    parent_prim: UsdGeom.Xform, 
    shape: TopoDS_Shape, 
    name: str
) -> Optional[UsdGeom.Mesh]:
    """Convert a triangulated shape to a USD Mesh with optional sharp edge detection and creases.
    
    Args:
        stage: USD stage
        parent_prim: Parent USD prim
        shape: OpenCASCADE shape to convert
        name: Name for the mesh
        
    Returns:
        Created USD Mesh or None if conversion failed
    """
    
    triangulate_shape(shape)
    faces: List[FaceData] =  extract_faces(shape)
    
    if not faces:
        print(f"    Warning: No triangulated faces found for shape {name}")
        return None
    
    # Analyze edges for sharp edge detection if enabled
    sharp_edges = []
    if ENABLE_SHARP_EDGE_DETECTION:
        try:
            print(f"    🔍 Analyzing edges for shape {name}...")
            all_edges, sharp_edges = analyze_step_edges(shape)
            print(f"    📊 Found {len(sharp_edges)} sharp edges out of {len(all_edges)} total")
        except Exception as e:
            print(f"    Warning: Sharp edge analysis failed for {name}: {e}")
            sharp_edges = []
    
    # Generate mesh data with face-varying normals if sharp edges detected
    if USE_FACE_VARYING_NORMALS and sharp_edges:
        mesh = _create_mesh_with_face_varying_normals(
            stage, parent_prim, shape, name, faces, sharp_edges
        )
    else:
        mesh = _create_mesh_with_vertex_normals(
            stage, parent_prim, shape, name, faces
        )
    
    # Add USD creases if enabled and we have a mesh
    if ENABLE_USD_CREASES and mesh and sharp_edges:
        try:
            _add_usd_creases_to_mesh(mesh, faces, sharp_edges)
        except Exception as e:
            print(f"    Warning: Failed to add USD creases to {name}: {e}")
    
    return mesh


def _create_mesh_with_face_varying_normals(
    stage: Usd.Stage,
    parent_prim: UsdGeom.Xform,
    shape: TopoDS_Shape,
    name: str,
    faces: List[FaceData],
    sharp_edges: List[Any]
) -> Optional[UsdGeom.Mesh]:
    """Create USD mesh with face-varying normals for sharp edges."""
    
    try:
        # Generate face-varying mesh data
        all_normals, all_face_vertex_counts, all_face_vertex_indices = generate_face_varying_normals(
            faces, sharp_edges
        )
        
        # Generate face-varying UV coordinates if enabled
        all_uv_coords = []
        if ENABLE_UV_COORDINATES:
            try:
                all_uv_coords, _, _ = generate_face_varying_uv_coordinates(faces, sharp_edges)
                print(f"    📐 Generated {len(all_uv_coords)} UV coordinates")
            except Exception as e:
                print(f"    ⚠️ Warning: UV coordinate generation failed: {e}")
                all_uv_coords = []
        
        # Build points list - for face-varying, each triangle gets unique vertices
        all_points: List[Point3D] = []
        vertex_index = 0
        
        for face, triangulation in faces:
            if triangulation is None:
                continue
                
            nb_triangles = triangulation.NbTriangles()
            
            for tri_idx in range(1, nb_triangles + 1):
                triangle = triangulation.Triangle(tri_idx)
                n1, n2, n3 = triangle.Get()
                
                # Get triangle vertices
                p1 = triangulation.Node(n1)
                p2 = triangulation.Node(n2)
                p3 = triangulation.Node(n3)
                
                # Add unique points for this triangle (face-varying)
                all_points.extend([
                    (float(p1.X()), float(p1.Y()), float(p1.Z())),
                    (float(p2.X()), float(p2.Y()), float(p2.Z())),
                    (float(p3.X()), float(p3.Y()), float(p3.Z()))
                ])
        
        if not all_points:
            print(f"    Warning: No vertices generated for shape {name}")
            return None
        
        # Apply winding order correction if enabled
        from .calculate_normals import ensure_consistent_winding_order
        if FORCE_CONSISTENT_WINDING and all_normals:
            all_face_vertex_indices, all_normals = ensure_consistent_winding_order(
                all_points, all_face_vertex_indices, all_normals
            )

        # Create USD mesh
        mesh_path: Sdf.Path = parent_prim.GetPath().AppendChild(name)
        mesh: UsdGeom.Mesh = UsdGeom.Mesh.Define(stage, mesh_path)

        # Set mesh attributes
        mesh.CreatePointsAttr().Set(all_points)
        mesh.CreateFaceVertexCountsAttr().Set(all_face_vertex_counts)
        mesh.CreateFaceVertexIndicesAttr().Set(all_face_vertex_indices)

        if all_normals:
            mesh.CreateNormalsAttr().Set(all_normals)
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        
        # Add UV coordinates as primvar if available
        if ENABLE_UV_COORDINATES and all_uv_coords:
            try:
                # Create UV primvar for texture mapping using PrimvarsAPI
                primvars_api = UsdGeom.PrimvarsAPI(mesh)
                uv_primvar = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
                uv_primvar.Set(all_uv_coords)
                print(f"    ✅ Added UV coordinates as 'st' primvar ({len(all_uv_coords)} values)")
            except Exception as e:
                print(f"    ⚠️ Warning: Failed to set UV primvar: {e}")

        print(f"    ✅ Created face-varying mesh: {len(all_points)} vertices, {len(all_face_vertex_counts)} faces")
        
        # Add creases if enabled (NEW FUNCTIONALITY)
        if ENABLE_USD_CREASES and sharp_edges:
            try:
                _add_usd_creases_to_mesh(mesh, faces, sharp_edges)
            except Exception as e:
                print(f"    Warning: Failed to add creases to face-varying mesh: {e}")
        
        return mesh
        
    except Exception as e:
        print(f"    Warning: Face-varying mesh creation failed for {name}: {e}")
        return None

def _create_mesh_with_vertex_normals(
    stage: Usd.Stage,
    parent_prim: UsdGeom.Xform,
    shape: TopoDS_Shape,
    name: str,
    faces: List[FaceData]
) -> Optional[UsdGeom.Mesh]:
    """Create USD mesh with traditional vertex normals (original implementation)."""
    
    all_points: List[Point3D] = []
    all_face_vertex_counts: List[int] = []
    all_face_vertex_indices: List[int] = []
    all_normals: List[Normal3D] = []
    vertex_map: VertexMap = {}
    next_vertex_index: int = 0
    
    for face, triangulation in faces:
        if triangulation is None:
            continue
        
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

    # Add UV coordinates if enabled
    if ENABLE_UV_COORDINATES:
        try:
            # Generate vertex-based UV coordinates for this mesh type
            all_uv_coords = []
            for face, triangulation in faces:
                if triangulation is None:
                    continue
                
                face_uvs, _ = calculate_parametric_uv_coordinates(face, triangulation, None, None)
                all_uv_coords.extend(face_uvs)
            
            if all_uv_coords:
                # Create UV primvar for texture mapping using PrimvarsAPI
                primvars_api = UsdGeom.PrimvarsAPI(mesh)
                uv_primvar = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
                uv_primvar.Set(all_uv_coords)
                print(f"    ✅ Added UV coordinates as 'st' primvar ({len(all_uv_coords)} values)")
                
        except Exception as e:
            print(f"    ⚠️ Warning: UV coordinate generation failed: {e}")

    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    
    # Add creases if enabled and we detected sharp edges (NEW FUNCTIONALITY)
    if ENABLE_USD_CREASES:
        try:
            # Re-run edge analysis just for crease generation if we didn't do it before
            if not ENABLE_SHARP_EDGE_DETECTION:
                print(f"    🔍 Running edge analysis for crease generation on {name}...")
                all_edges, detected_sharp_edges = analyze_step_edges(shape)
                if detected_sharp_edges:
                    _add_usd_creases_to_mesh(mesh, faces, detected_sharp_edges)
            # If we already have sharp edge detection enabled, creases were added in convert_shape_to_usd_mesh
        except Exception as e:
            print(f"    Warning: Failed to add creases to vertex normal mesh: {e}")
    
    return mesh


def _add_usd_creases_to_mesh(
    mesh: UsdGeom.Mesh,
    faces: List[FaceData], 
    sharp_edges: List[Any]
) -> None:
    """Add USD crease attributes to an existing mesh based on sharp edge analysis.
    
    This function adds standard USD crease attributes (creaseIndices, creaseLengths, 
    creaseSharpnesses) to enable proper sharp edge rendering in USD-aware renderers.
    
    Args:
        mesh: USD mesh to add creases to
        faces: List of face data used to build the mesh
        sharp_edges: List of detected sharp edges
    """
    print("    🎯 Adding USD creases to mesh...")
    
    try:
        # Get the current mesh points for vertex mapping
        points_attr = mesh.GetPointsAttr()
        mesh_points = points_attr.Get()
        
        if not mesh_points:
            print("    Warning: No mesh points available for crease generation")
            return
        
        # Generate crease data from sharp edges
        crease_data = generate_usd_creases_from_edges(faces, sharp_edges, mesh_points)
        
        if not crease_data.has_creases():
            print("    No valid creases generated from sharp edges")
            return
        
        # Set USD crease attributes
        mesh.CreateCreaseIndicesAttr().Set(crease_data.edge_indices)
        mesh.CreateCreaseLengthsAttr().Set([2] * len(crease_data.edge_sharpnesses))  # Each edge uses 2 indices
        mesh.CreateCreaseSharpnessesAttr().Set(crease_data.edge_sharpnesses)
        
        num_creases = len(crease_data.edge_sharpnesses)
        print(f"    ✅ Added {num_creases} USD creases to mesh")
        
    except Exception as e:
        print(f"    Warning: Failed to add USD creases: {e}")


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
    
    
    # Clean up very small values to avoid -0.0 display issues
    def clean_float(val: float) -> float:
        return 0.0 if abs(val) < 1e-10 else float(val)
    
    #MUST LEAVE THE MATRIX TRANPOSED FUCK THIS SHIT

    return Gf.Matrix4d(
        clean_float(trsf.Value(1, 1)), clean_float(trsf.Value(2, 1)), clean_float(trsf.Value(3, 1)), 0.0,
        clean_float(trsf.Value(1, 2)), clean_float(trsf.Value(2, 2)), clean_float(trsf.Value(3, 2)), 0.0,
        clean_float(trsf.Value(1, 3)), clean_float(trsf.Value(2, 3)), clean_float(trsf.Value(3, 3)), 0.0,
        clean_float(translation.X()),  clean_float(translation.Y()),  clean_float(translation.Z()), 1.0
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

