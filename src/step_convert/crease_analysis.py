
from __future__ import annotations
from typing import List, Tuple, Dict, Set, Optional, Any


from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Edge
from OCC.Core.Poly import Poly_Triangulation
from OCC.Core.gp import gp_Pnt

from .edge_analysis import EdgeInfo


class CreaseData:
    """Data structure for USD crease information."""
    
    def __init__(self):
        self.edge_indices: List[int] = []  # Pairs of vertex indices defining edges
        self.edge_sharpnesses: List[float] = []  # Sharpness values for each edge
        
    def add_edge(self, vertex1_idx: int, vertex2_idx: int, sharpness: float = 1.0):
        """Add a crease edge with specified sharpness.
        
        Args:
            vertex1_idx: First vertex index
            vertex2_idx: Second vertex index
            sharpness: Edge sharpness (0.0 = smooth, 1.0 = infinitely sharp)
        """
        self.edge_indices.extend([vertex1_idx, vertex2_idx])
        self.edge_sharpnesses.append(sharpness)
    
    def has_creases(self) -> bool:
        """Check if any creases have been defined."""
        return len(self.edge_indices) > 0



def generate_usd_creases_from_edges(
    faces: List[Tuple[TopoDS_Face, Optional[Poly_Triangulation]]], 
    sharp_edges: List[EdgeInfo],
    mesh_points: List[Tuple[float, float, float]]
) -> CreaseData:
    """Generate USD crease data from detected sharp edges.
    
    This function maps the sharp edges detected in the geometric analysis
    to USD crease format, which requires vertex index pairs and sharpness values.
    
    Args:
        faces: List of (face, triangulation) tuples
        sharp_edges: List of detected sharp edges from edge analysis
        mesh_points: List of mesh vertex positions for index mapping
        
    Returns:
        CreaseData containing edge indices and sharpness values for USD
    """
    print("🎯 Generating USD creases from detected sharp edges...")
    
    crease_data = CreaseData()
    
    if not sharp_edges:
        print("    No sharp edges detected - no creases to generate")
        return crease_data
    
    # Build a spatial index of mesh vertices for efficient lookup
    vertex_spatial_map = _build_vertex_spatial_map(mesh_points)
    
    # Process each sharp edge
    processed_edges = 0
    for edge_info in sharp_edges:
        try:
            # Extract edge geometry
            edge_vertices = _extract_edge_vertices(edge_info.edge)
            
            if len(edge_vertices) < 2:
                continue
            
            # Map edge vertices to mesh vertex indices
            mesh_vertex_indices = []
            for edge_vertex in edge_vertices:
                vertex_point = (float(edge_vertex.X()), float(edge_vertex.Y()), float(edge_vertex.Z()))
                mesh_idx = _find_closest_mesh_vertex(vertex_point, vertex_spatial_map, mesh_points)
                if mesh_idx is not None:
                    mesh_vertex_indices.append(mesh_idx)
            
            # Create crease edges from consecutive vertex pairs
            for i in range(len(mesh_vertex_indices) - 1):
                v1_idx = mesh_vertex_indices[i]
                v2_idx = mesh_vertex_indices[i + 1]
                
                # Calculate sharpness based on edge characteristics
                sharpness = _calculate_edge_sharpness(edge_info)
                
                crease_data.add_edge(v1_idx, v2_idx, sharpness)
                processed_edges += 1
                
        except Exception as e:
            print(f"    Warning: Failed to process sharp edge: {e}")
            continue
    
    print(f"    ✅ Generated {processed_edges} crease edges from {len(sharp_edges)} sharp edges")
    return crease_data


def _build_vertex_spatial_map(mesh_points: List[Tuple[float, float, float]]) -> Dict[Tuple[int, int, int], List[int]]:
    """Build a spatial hash map for efficient vertex lookup.
    
    Args:
        mesh_points: List of mesh vertex positions
        
    Returns:
        Dictionary mapping spatial cells to vertex indices
    """
    spatial_map = {}
    cell_size = 0.001 
    
    for idx, (x, y, z) in enumerate(mesh_points):
        cell_x = int(x / cell_size)
        cell_y = int(y / cell_size)
        cell_z = int(z / cell_size)
        cell_key = (cell_x, cell_y, cell_z)
        
        if cell_key not in spatial_map:
            spatial_map[cell_key] = []
        spatial_map[cell_key].append(idx)
    
    return spatial_map


def _extract_edge_vertices(edge: TopoDS_Edge) -> List[gp_Pnt]:
    """Extract vertex positions along an edge.
    
    Args:
        edge: OpenCASCADE edge
        
    Returns:
        List of points along the edge
    """
    try:
        from OCC.Core.BRep import BRep_Tool
        
        edge_curve, first_param, last_param = BRep_Tool.Curve(edge)
        if edge_curve is None:
            return []
        
        num_samples = 5
        vertices = []
        
        for i in range(num_samples):
            if num_samples == 1:
                param = (first_param + last_param) / 2.0
            else:
                param = first_param + (last_param - first_param) * i / (num_samples - 1)
            
            point = gp_Pnt()
            edge_curve.D0(param, point)
            vertices.append(point)
        
        return vertices
        
    except Exception:
        return []


def _find_closest_mesh_vertex(
    target_point: Tuple[float, float, float], 
    spatial_map: Dict[Tuple[int, int, int], List[int]], 
    mesh_points: List[Tuple[float, float, float]],
    tolerance: float = 0.01
) -> Optional[int]:
    """Find the closest mesh vertex to a target point using spatial hashing.
    
    Args:
        target_point: Target vertex position
        spatial_map: Spatial hash map of vertices
        mesh_points: List of mesh vertex positions
        tolerance: Maximum distance tolerance for matching
        
    Returns:
        Index of closest mesh vertex, or None if no match found
    """
    try:
        x, y, z = target_point
        cell_size = 0.001  # Must match the cell size used in _build_vertex_spatial_map
        
        # Check the target cell and neighboring cells
        target_cell_x = int(x / cell_size)
        target_cell_y = int(y / cell_size)
        target_cell_z = int(z / cell_size)
        
        best_distance = float('inf')
        best_index = None
        
        # Search in a 3x3x3 neighborhood around the target cell
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    cell_key = (target_cell_x + dx, target_cell_y + dy, target_cell_z + dz)
                    
                    if cell_key in spatial_map:
                        for vertex_idx in spatial_map[cell_key]:
                            mesh_point = mesh_points[vertex_idx]
                            distance = _distance_3d(target_point, mesh_point)
                            
                            if distance < best_distance and distance <= tolerance:
                                best_distance = distance
                                best_index = vertex_idx
        
        return best_index
        
    except Exception:
        return None


def _distance_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """Calculate 3D distance between two points."""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**0.5


def _calculate_edge_sharpness(edge_info: EdgeInfo) -> float:
    """Calculate USD sharpness value from edge analysis.
    
    Args:
        edge_info: Edge information from sharp edge detection
        
    Returns:
        Sharpness value (0.0 = smooth, 1.0 = infinitely sharp)
    """
    if not edge_info.is_sharp:
        return 0.0

    angle : float = min(90, 180 - edge_info.angle)

    return angle / 9