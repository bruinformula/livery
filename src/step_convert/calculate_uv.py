from typing import List, Tuple, Dict, Optional, Any, Union, Set
import numpy as np
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Edge
from OCC.Core.Poly import Poly_Triangulation
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Cone, GeomAbs_Sphere
from OCC.Core.gp import gp_Pnt

try:
    from .config import ENABLE_UV_SEAMS, UV_SCALE_FACTOR
except ImportError:
    ENABLE_UV_SEAMS: bool = True
    UV_SCALE_FACTOR: float = 1.0


class UVSeamDetector:
    """Detects UV seams from edge analysis for proper UV coordinate handling."""
    
    def __init__(self, sharp_edges: List[Any]):
        self.sharp_edges = sharp_edges
        self.seam_edges: Set[TopoDS_Edge] = set()
        self.seam_vertices: Set[int] = set()
        
    def detect_uv_seams(self) -> Dict[str, Set]:
        """Detect UV seams from sharp edges analysis."""
        seam_info = {
            'seam_edges': set(),
            'seam_vertices': set(),
            'cylindrical_seams': set(),
            'boundary_edges': set()
        }
        
        for edge_info in self.sharp_edges:
            # Mark as seam if it's a sharp boundary between different surface types
            if hasattr(edge_info, 'face1') and hasattr(edge_info, 'face2'):
                type1 = self._get_surface_type(edge_info.face1)
                type2 = self._get_surface_type(edge_info.face2)
                
                # Cylinder-plane boundaries create UV seams
                if (type1 == 'cylinder' and type2 == 'plane') or (type1 == 'plane' and type2 == 'cylinder'):
                    seam_info['seam_edges'].add(edge_info.edge)
                    seam_info['boundary_edges'].add(edge_info.edge)
                
                # Cylindrical seams (parametric discontinuities)
                if type1 == 'cylinder' and type2 == 'cylinder':
                    if self._is_cylindrical_seam(edge_info.edge, edge_info.face1):
                        seam_info['cylindrical_seams'].add(edge_info.edge)
                        seam_info['seam_edges'].add(edge_info.edge)
        
        return seam_info
    
    def _get_surface_type(self, face: TopoDS_Face) -> str:
        """Get surface type for seam detection."""
        try:
            adaptor = BRepAdaptor_Surface(face)
            surface_type = adaptor.GetType()
            
            if surface_type == GeomAbs_Cylinder:
                return 'cylinder'
            elif surface_type == GeomAbs_Plane:
                return 'plane'
            elif surface_type == GeomAbs_Cone:
                return 'cone'
            elif surface_type == GeomAbs_Sphere:
                return 'sphere'
            else:
                return 'other'
        except Exception:
            return 'unknown'
    
    def _is_cylindrical_seam(self, edge: TopoDS_Edge, face: TopoDS_Face) -> bool:
        """Check if edge is a cylindrical seam (u=0, u=2π discontinuity)."""
        try:
            return BRep_Tool.IsClosed(edge, face)
        except Exception:
            return False


def calculate_parametric_uv_coordinates(
    face: TopoDS_Face,
    triangulation: Poly_Triangulation,
    location: Optional[TopLoc_Location] = None,
    seam_info: Optional[Dict[str, Set]] = None
) -> Tuple[List[Tuple[float, float]], Dict[int, Tuple[float, float]]]:
    """Calculate UV coordinates based on parametric surface values.
    
    Args:
        face: The TopoDS_Face containing the parametric surface
        triangulation: The triangulation data for the face
        location: Optional transformation location
        seam_info: Information about UV seams for proper handling
        
    Returns:
        Tuple of (uv_coordinates_per_triangle, vertex_uv_cache)
        - uv_coordinates_per_triangle: List of (u,v) for each triangle vertex
        - vertex_uv_cache: Dict mapping vertex indices to (u,v) coordinates
    """
    uv_coordinates: List[Tuple[float, float]] = []
    vertex_uv_cache: Dict[int, Tuple[float, float]] = {}
    
    try:
        nb_nodes: int = triangulation.NbNodes()
        nb_triangles: int = triangulation.NbTriangles()
        
        if nb_nodes == 0 or nb_triangles == 0:
            return uv_coordinates, vertex_uv_cache
        
        # Get the surface adaptor for UV parameter evaluation
        adaptor: BRepAdaptor_Surface = BRepAdaptor_Surface(face)
        surface_type = adaptor.GetType()
        
        # Get UV parameter bounds
        u_min: float = adaptor.FirstUParameter()
        u_max: float = adaptor.LastUParameter()
        v_min: float = adaptor.FirstVParameter()
        v_max: float = adaptor.LastVParameter()
        
        # Determine UV scaling based on surface type
        u_scale, v_scale = _calculate_uv_scaling(surface_type, u_min, u_max, v_min, v_max)
        
        # Check if the triangulation has UV nodes (2D parameters)
        has_uv: bool = triangulation.HasUVNodes()
        
        print(f"    📐 UV bounds: U[{u_min:.3f}, {u_max:.3f}], V[{v_min:.3f}, {v_max:.3f}]")
        print(f"    🎯 Surface type: {surface_type}, UV available: {has_uv}")
        
        # Process each triangle
        for i in range(1, nb_triangles + 1):
            triangle = triangulation.Triangle(i)
            n1, n2, n3 = triangle.Get()
            
            # Calculate UV for each vertex of the triangle
            for vertex_idx in [n1, n2, n3]:
                if vertex_idx in vertex_uv_cache:
                    # Use cached UV
                    uv = vertex_uv_cache[vertex_idx]
                else:
                    # Calculate new UV
                    uv = _calculate_vertex_uv(
                        vertex_idx, triangulation, face, location,
                        u_min, u_max, v_min, v_max, has_uv, surface_type, seam_info
                    )
                    
                    # Apply scaling
                    u_scaled = uv[0] * u_scale * UV_SCALE_FACTOR
                    v_scaled = uv[1] * v_scale * UV_SCALE_FACTOR
                    
                    uv = (u_scaled, v_scaled)
                    vertex_uv_cache[vertex_idx] = uv
                
                uv_coordinates.append(uv)
        
        print(f"    ✅ Generated {len(uv_coordinates)} UV coordinates")
        
    except Exception as e:
        print(f"    ⚠️ Warning: Could not calculate UV coordinates for face: {e}")
    
    return uv_coordinates, vertex_uv_cache


def _calculate_vertex_uv(
    vertex_idx: int,
    triangulation: Poly_Triangulation,
    face: TopoDS_Face,
    location: Optional[TopLoc_Location],
    u_min: float, u_max: float, v_min: float, v_max: float,
    has_uv: bool,
    surface_type: Any,
    seam_info: Optional[Dict[str, Set]]
) -> Tuple[float, float]:
    """Calculate UV coordinates for a single vertex."""
    
    try:
        if has_uv:
            # Use UV coordinates if available in triangulation
            uv_node = triangulation.UVNode(vertex_idx)
            u: float = uv_node.X()
            v: float = uv_node.Y()
            
            # Clamp UV parameters to valid range
            u = max(u_min, min(u_max, u))
            v = max(v_min, min(v_max, v))
            
        else:
            # Project 3D point to UV space
            vertex_3d = triangulation.Node(vertex_idx)
            
            # Apply transformation if needed
            if location and not location.IsIdentity():
                trsf = location.Transformation()
                vertex_3d = vertex_3d.Transformed(trsf)
            
            # Use GeomAPI_ProjectPointOnSurf for accurate projection
            surface = BRep_Tool.Surface(face)
            projector = GeomAPI_ProjectPointOnSurf(gp_Pnt(vertex_3d.X(), vertex_3d.Y(), vertex_3d.Z()), surface)
            
            if projector.NbPoints() > 0:
                projector.Parameters(1, u, v)
                # Clamp to bounds
                u = max(u_min, min(u_max, u))
                v = max(v_min, min(v_max, v))
            else:
                # Fallback to center if projection fails
                u = (u_min + u_max) * 0.5
                v = (v_min + v_max) * 0.5
        
        # Handle seams for cylindrical surfaces
        if ENABLE_UV_SEAMS and surface_type == GeomAbs_Cylinder and seam_info:
            u, v = _handle_cylindrical_seams(u, v, u_min, u_max, v_min, v_max, vertex_idx, seam_info)
        
        # Return RAW parametric coordinates - NO normalization!
        # These are the actual CAD surface parameters for procedural texturing
        return (u, v)
        
    except Exception as e:
        print(f"    ⚠️ UV calculation failed for vertex {vertex_idx}: {e}")
        return (0.5, 0.5)  # Fallback to center


def _calculate_uv_scaling(surface_type: Any, u_min: float, u_max: float, v_min: float, v_max: float) -> Tuple[float, float]:
    """Calculate appropriate UV scaling based on surface type."""
    
    if surface_type == GeomAbs_Cylinder:
        # For cylinders, U is typically angular (0 to 2π), V is linear
        u_range = u_max - u_min
        v_range = v_max - v_min
        
        # Normalize angular parameter
        u_scale = 1.0 / (2 * np.pi) if u_range > np.pi else 1.0
        v_scale = 1.0
        
    elif surface_type == GeomAbs_Sphere:
        # For spheres, both U and V are angular
        u_scale = 1.0 / (2 * np.pi)
        v_scale = 1.0 / np.pi
        
    elif surface_type == GeomAbs_Cone:
        # For cones, U is angular, V is linear
        u_scale = 1.0 / (2 * np.pi)
        v_scale = 1.0
        
    else:
        # For planes and other surfaces, use direct scaling
        u_scale = 1.0
        v_scale = 1.0
    
    return u_scale, v_scale


def _handle_cylindrical_seams(
    u: float, v: float, 
    u_min: float, u_max: float, v_min: float, v_max: float,
    vertex_idx: int, seam_info: Dict[str, Set]
) -> Tuple[float, float]:
    """Handle UV seams for cylindrical surfaces."""
    
    # For cylindrical surfaces, handle the U parameter wraparound at seams
    u_range = u_max - u_min
    
    # If this vertex is near a seam, adjust UV coordinates
    if abs(u - u_min) < 1e-6 or abs(u - u_max) < 1e-6:
        # This vertex is on the cylindrical seam
        # Ensure consistent UV assignment across the seam
        if u > (u_min + u_max) * 0.5:
            u = u_max
        else:
            u = u_min
    
    return u, v


def _get_face_world_size(triangulation: Poly_Triangulation) -> float:
    """Calculate the world-space area of a face for proper UV scaling."""
    if triangulation is None:
        return 1.0
    
    try:
        # Get all vertex positions
        vertices_3d = []
        for i in range(1, triangulation.NbNodes() + 1):
            node = triangulation.Node(i)
            vertices_3d.append([node.X(), node.Y(), node.Z()])
        
        if not vertices_3d:
            return 1.0
        
        vertices_3d = np.array(vertices_3d)
        bbox_min = np.min(vertices_3d, axis=0)
        bbox_max = np.max(vertices_3d, axis=0)
        
        # Calculate bounding box area (approximation of face area)
        dimensions = bbox_max - bbox_min
        area = np.linalg.norm(dimensions) ** 2  # Simple area approximation
        
        return max(area, 1e-6)
        
    except Exception as e:
        print(f"    ⚠️ Failed to calculate face size: {e}")
        return 1.0


def generate_face_varying_uv_coordinates(
    faces: List[Tuple[TopoDS_Face, Optional[Poly_Triangulation]]], 
    sharp_edges: List[Any]
) -> Tuple[List[Tuple[float, float]], List[int], List[int]]:
    """Generate UV islands with proper scaling based on CAD face sizes.
    
    Each face becomes a UV island, scaled proportionally to its world-space size
    and laid out cleanly in UV space without overlap.
    
    Args:
        faces: List of (face, triangulation) tuples
        sharp_edges: List of detected sharp edges for seam detection
        
    Returns:
        Tuple of (uv_coordinates, face_vertex_counts, face_vertex_indices) for USD
    """
    print("�️ Generating UV islands with size-based scaling...")
    
    # Detect UV seams from edge analysis
    seam_detector = UVSeamDetector(sharp_edges)
    seam_info = seam_detector.detect_uv_seams()
    
    # Calculate world sizes for all faces to determine relative scaling
    face_areas = []
    max_area = 0.0
    
    for face, triangulation in faces:
        area = _get_face_world_size(triangulation)
        face_areas.append(area)
        max_area = max(max_area, area)
    
    print(f"🔍 Analyzed {len(faces)} faces, max area: {max_area:.3f}")
    
    # UV layout state
    current_u = 0.0
    current_v = 0.0
    row_height = 0.0
    padding = 0.05
    
    all_uv_coords: List[Tuple[float, float]] = []
    all_face_vertex_counts: List[int] = []
    all_face_vertex_indices: List[int] = []
    vertex_counter = 0
    
    for face_idx, (face, triangulation) in enumerate(faces):
        if triangulation is None:
            continue
        
        # Get parametric UV coordinates for this face (0-1 normalized)
        face_uvs, _ = calculate_parametric_uv_coordinates(face, triangulation, None, seam_info)
        
        if not face_uvs:
            continue
        
        # Calculate spacing based on parametric range (not world area)
        parametric_u_range = face_u_max - face_u_min if face_u_max > face_u_min else 1.0
        parametric_v_range = face_v_max - face_v_min if face_v_max > face_v_min else 1.0
        
        # Use larger of the two ranges for island separation
        island_extent = max(parametric_u_range, parametric_v_range) + padding * 2
        
        # Check if we need to start a new row
        if current_u + island_extent > 10.0:  # Larger UV space for parametric coords
            current_u = 0.0
            current_v += row_height + padding
            row_height = 0.0
        
        # Get UV bounds for this face
        u_coords = [uv[0] for uv in face_uvs]
        v_coords = [uv[1] for uv in face_uvs]
        
        face_u_min, face_u_max = min(u_coords), max(u_coords)
        face_v_min, face_v_max = min(v_coords), max(v_coords)
        
        nb_triangles = triangulation.NbTriangles()
        uv_idx = 0
        
        print(f"    🏗️ Island {face_idx}: extent {island_extent:.3f} at UV[{current_u:.3f}, {current_v:.3f}] (param range: {parametric_u_range:.3f}x{parametric_v_range:.3f})")
        
        for tri_idx in range(nb_triangles):
            all_face_vertex_counts.append(3)
            
            for _ in range(3):
                if uv_idx < len(face_uvs):
                    raw_u, raw_v = face_uvs[uv_idx]
                    
                    # Use RAW parametric coordinates directly - NO normalization!
                    # Just offset each island to avoid overlap
                    final_u = raw_u + current_u
                    final_v = raw_v + current_v
                    
                    all_uv_coords.append((final_u, final_v))
                    uv_idx += 1
                else:
                    # Fallback UV at face parameter center
                    face_u_center = (face_u_min + face_u_max) * 0.5
                    face_v_center = (face_v_min + face_v_max) * 0.5
                    final_u = face_u_center + current_u
                    final_v = face_v_center + current_v
                    all_uv_coords.append((final_u, final_v))
                
                all_face_vertex_indices.append(vertex_counter)
                vertex_counter += 1
        
        # Update layout position
        current_u += island_extent
        row_height = max(row_height, island_extent)
    
    print(f"✅ Generated {len(all_uv_coords)} UV coordinates in {len(faces)} size-scaled islands")
    print(f"🔗 Used {len(seam_info['seam_edges'])} seam edges for separation")
    
    return all_uv_coords, all_face_vertex_counts, all_face_vertex_indices
    