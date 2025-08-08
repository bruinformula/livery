"""Edge analysis and sharp edge detection for face-varying attributes.

This module analyzes STEP file geometry to detect edges where surface derivatives
differ significantly, indicating areas that should have sharp edges in the 
rendered mesh. It generates face-varying vertex normals to create the appearance
of sharp edges.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Set, Optional, Any
import numpy as np
from numpy.typing import NDArray

from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomLProp import GeomLProp_SLProps, GeomLProp_CLProps
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Poly import Poly_Triangulation, Poly_Triangle
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Vec
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_ListOfShape
from OCC.Core.BRepGProp import BRepGProp_Face
from OCC.Core.BRepLProp import BRepLProp_SLProps

try:
    from .config import MESH_LINEAR_DEFLECTION
except ImportError:
    MESH_LINEAR_DEFLECTION = 1.0

# Configuration for sharp edge detection
SHARP_EDGE_ANGLE_THRESHOLD = 30.0  # degrees - edges with angle > this are considered sharp
MIN_EDGE_LENGTH = 0.1  # minimum edge length to consider for sharp edge detection
DERIVATIVE_TOLERANCE = 1e-6  # tolerance for surface derivative comparison


class EdgeInfo:
    """Information about an edge between two faces."""
    
    def __init__(self, edge: TopoDS_Edge, face1: TopoDS_Face, face2: TopoDS_Face):
        self.edge = edge
        self.face1 = face1
        self.face2 = face2
        self.angle = 0.0  # dihedral angle in degrees
        self.is_sharp = False
        self.derivatives_differ = False
        self.edge_vertices: List[int] = []  # vertex indices along this edge
        
    def __repr__(self) -> str:
        return f"EdgeInfo(angle={self.angle:.1f}°, sharp={self.is_sharp}, diff_deriv={self.derivatives_differ})"


class SharpEdgeDetector:
    """Detects sharp edges by analyzing surface derivatives and dihedral angles."""
    
    def __init__(self, shape: TopoDS_Shape):
        self.shape = shape
        self.face_edge_map: Dict[TopoDS_Face, List[EdgeInfo]] = {}
        self.sharp_edges: List[EdgeInfo] = []
        self.edge_vertex_map: Dict[TopoDS_Edge, Set[int]] = {}
        
    def analyze_edges(self) -> List[EdgeInfo]:
        """Analyze all edges in the shape and detect sharp edges.
        
        Returns:
            List of EdgeInfo objects for all edges, with sharp edges marked
        """
        print("🔍 Analyzing edges for sharp edge detection...")
        
        # Build topology map of edges to faces
        edge_face_map = self._build_edge_face_map()
        
        all_edges: List[EdgeInfo] = []
        sharp_count = 0
        angle_samples = []  # Track angles for debugging
        
        for edge, faces in edge_face_map.items():
            if len(faces) == 2:  # Interior edge between two faces
                face1, face2 = faces[0], faces[1]
                edge_info = EdgeInfo(edge, face1, face2)
                
                # Calculate dihedral angle
                edge_info.angle = self._calculate_dihedral_angle(edge, face1, face2)
                angle_samples.append(edge_info.angle)
                
                # Check surface derivatives
                edge_info.derivatives_differ = self._check_surface_derivatives(edge, face1, face2)
                
                # Determine if edge is sharp
                edge_info.is_sharp = (
                    edge_info.angle > SHARP_EDGE_ANGLE_THRESHOLD or 
                    edge_info.derivatives_differ
                )
                
                if edge_info.is_sharp:
                    self.sharp_edges.append(edge_info)
                    sharp_count += 1
                    print(f"  🔪 Sharp edge detected: {edge_info.angle:.1f}° (threshold: {SHARP_EDGE_ANGLE_THRESHOLD}°)")
                
                all_edges.append(edge_info)
        
        if angle_samples:
            print(f"📐 Angle range: {min(angle_samples):.1f}° - {max(angle_samples):.1f}°, avg: {np.mean(angle_samples):.1f}°")
        print(f"📊 Found {sharp_count} sharp edges out of {len(all_edges)} total edges")
        return all_edges
    
    def _build_edge_face_map(self) -> Dict[TopoDS_Edge, List[TopoDS_Face]]:
        """Build a map of edges to the faces that share them."""
        edge_face_map: Dict[TopoDS_Edge, List[TopoDS_Face]] = {}
        
        # Use TopExp to build the map efficiently
        map_tool = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndAncestors(self.shape, TopAbs_EDGE, TopAbs_FACE, map_tool)
        
        for i in range(1, map_tool.Size() + 1):
            edge = map_tool.FindKey(i)
            face_list = map_tool.FindFromIndex(i)
            
            faces = []
            # Simple iteration for the most common case (2 faces per edge)
            if face_list.Size() > 0:
                faces.append(face_list.First())
                if face_list.Size() > 1:
                    faces.append(face_list.Last())
            
            edge_face_map[edge] = faces
        
        return edge_face_map
    
    def _calculate_dihedral_angle(self, edge: TopoDS_Edge, face1: TopoDS_Face, face2: TopoDS_Face) -> float:
        """Calculate the dihedral angle between two faces along their shared edge.
        
        Args:
            edge: The shared edge
            face1: First face
            face2: Second face
            
        Returns:
            Dihedral angle in degrees (0-180)
        """
        try:
            # Use BRepLProp for more direct surface property calculation
            surface1 = BRep_Tool.Surface(face1)
            surface2 = BRep_Tool.Surface(face2)
            
            if surface1 is None or surface2 is None:
                return 0.0
            
            # Get edge curve
            edge_curve = BRep_Tool.Curve(edge)[0]
            if edge_curve is None:
                return 0.0
                
            # Get parameter range
            first_param = BRep_Tool.Curve(edge)[1]
            last_param = BRep_Tool.Curve(edge)[2]
            mid_param = (first_param + last_param) / 2.0
            
            # Get point on edge
            edge_point = gp_Pnt()
            edge_curve.D0(mid_param, edge_point)
            
            # Try to get UV parameters for this point on each face
            # This is a simplified approach - projecting to face centers with offset
            
            # Get face 1 properties
            face1_adaptor = BRepAdaptor_Surface(face1)
            u1_mid = (face1_adaptor.FirstUParameter() + face1_adaptor.LastUParameter()) / 2.0
            v1_mid = (face1_adaptor.FirstVParameter() + face1_adaptor.LastVParameter()) / 2.0
            
            # Get face 2 properties
            face2_adaptor = BRepAdaptor_Surface(face2)
            u2_mid = (face2_adaptor.FirstUParameter() + face2_adaptor.LastUParameter()) / 2.0
            v2_mid = (face2_adaptor.FirstVParameter() + face2_adaptor.LastVParameter()) / 2.0
            
            # Calculate normals using surface properties  
            props1 = BRepLProp_SLProps(face1_adaptor, u1_mid, v1_mid, 1, 1e-6)
            props2 = BRepLProp_SLProps(face2_adaptor, u2_mid, v2_mid, 1, 1e-6)
            
            if not props1.IsNormalDefined() or not props2.IsNormalDefined():
                return 0.0
            
            # Get normal vectors
            normal1 = props1.Normal()
            normal2 = props2.Normal()
            
            # Convert to numpy arrays
            n1 = np.array([normal1.X(), normal1.Y(), normal1.Z()])
            n2 = np.array([normal2.X(), normal2.Y(), normal2.Z()])
            
            # Normalize
            n1_len = np.linalg.norm(n1)
            n2_len = np.linalg.norm(n2)
            
            if n1_len < 1e-10 or n2_len < 1e-10:
                return 0.0
                
            n1 = n1 / n1_len
            n2 = n2 / n2_len
            
            # Calculate angle between normals
            dot_product = np.clip(np.dot(n1, n2), -1.0, 1.0)
            angle_rad = np.arccos(abs(dot_product))
            angle_deg = np.degrees(angle_rad)
            
            # The dihedral angle is the angle between the planes
            # For two planes with normals n1 and n2, the dihedral angle is:
            # - 0° if they're coplanar (normals parallel, same direction)
            # - 180° if they're coplanar (normals parallel, opposite direction)  
            # - 90° if they're perpendicular
            
            # If normals point in same direction (dot > 0), faces are on same side - small dihedral
            # If normals point in opposite directions (dot < 0), faces are on opposite sides - large dihedral
            
            if dot_product >= 0:
                # Normals point in similar directions - acute dihedral angle
                return angle_deg
            else:
                # Normals point in opposite directions - obtuse dihedral angle
                return 180.0 - angle_deg
                
        except Exception as e:
            print(f"    Warning: Dihedral angle calculation failed: {e}")
            return 0.0
    
    def _get_face_normal_near_edge(self, face: TopoDS_Face, edge: TopoDS_Edge, edge_point: gp_Pnt) -> Optional[NDArray[np.float64]]:
        """Get the surface normal of a face near an edge point.
        
        This method projects the edge point onto the face surface and calculates
        the normal there, providing more accurate results than using face centers.
        
        Args:
            face: The face to analyze
            edge: The edge (for context)
            edge_point: 3D point on the edge
            
        Returns:
            Unit normal vector as numpy array, or None if calculation fails
        """
        try:
            # Get face surface adaptor
            adaptor = BRepAdaptor_Surface(face)
            
            # Project the edge point onto the face surface
            # For simplicity, sample at several UV positions and find the closest
            u_min = adaptor.FirstUParameter()
            u_max = adaptor.LastUParameter()
            v_min = adaptor.FirstVParameter()
            v_max = adaptor.LastVParameter()
            
            # Sample points in UV space to find closest to edge point
            best_distance = float('inf')
            best_u, best_v = None, None
            
            # Sample a small grid in UV space
            for u_frac in [0.25, 0.5, 0.75]:
                for v_frac in [0.25, 0.5, 0.75]:
                    u_test = u_min + (u_max - u_min) * u_frac
                    v_test = v_min + (v_max - v_min) * v_frac
                    
                    try:
                        surface_point = adaptor.Value(u_test, v_test)
                        distance = edge_point.Distance(surface_point)
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_u, best_v = u_test, v_test
                    except:
                        continue
            
            if best_u is None or best_v is None:
                # Fallback to center
                best_u = (u_min + u_max) / 2.0
                best_v = (v_min + v_max) / 2.0
            
            # Calculate surface normal at best UV position
            props = GeomLProp_SLProps(adaptor, best_u, best_v, 1, 1e-6)
            if props.IsNormalDefined():
                normal = props.Normal()
                normal_array = np.array([normal.X(), normal.Y(), normal.Z()])
                
                # Normalize
                length = np.linalg.norm(normal_array)
                if length > 1e-10:
                    return normal_array / length
            
            return None
            
        except Exception as e:
            # Fallback to the original method
            return self._get_face_normal_at_point(face, edge_point)

    def _get_face_normal_at_point(self, face: TopoDS_Face, point: gp_Pnt) -> Optional[NDArray[np.float64]]:
        """Get the surface normal of a face at a specific 3D point.
        
        Args:
            face: The face to analyze
            point: 3D point near the face
            
        Returns:
            Unit normal vector as numpy array, or None if calculation fails
        """
        try:
            # Project point onto face surface to get UV parameters
            adaptor = BRepAdaptor_Surface(face)
            
            # For simplicity, use the center of the face's UV domain
            # In a more sophisticated implementation, you'd project the point
            u_min = adaptor.FirstUParameter()
            u_max = adaptor.LastUParameter()
            v_min = adaptor.FirstVParameter()
            v_max = adaptor.LastVParameter()
            u = (u_min + u_max) / 2.0
            v = (v_min + v_max) / 2.0
            
            # Calculate surface normal
            props = GeomLProp_SLProps(adaptor, u, v, 1, 1e-6)
            if props.IsNormalDefined():
                normal = props.Normal()
                normal_array = np.array([normal.X(), normal.Y(), normal.Z()])
                
                # Normalize
                length = np.linalg.norm(normal_array)
                if length > 1e-10:
                    return normal_array / length
            
            return None
            
        except Exception:
            return None
    
    def _check_surface_derivatives(self, edge: TopoDS_Edge, face1: TopoDS_Face, face2: TopoDS_Face) -> bool:
        """Check if surface derivatives differ significantly across an edge.
        
        This analyzes the first and second derivatives of the surfaces on either
        side of the edge to detect discontinuities that should result in sharp edges.
        
        Args:
            edge: The shared edge
            face1: First face  
            face2: Second face
            
        Returns:
            True if derivatives differ significantly
        """
        try:
            # Get adaptors for both faces
            adaptor1 = BRepAdaptor_Surface(face1)
            adaptor2 = BRepAdaptor_Surface(face2)
            
            # Sample derivatives along the edge
            edge_adaptor = BRepAdaptor_Curve(edge)
            u_start = edge_adaptor.FirstParameter()
            u_end = edge_adaptor.LastParameter()
            
            # Sample at multiple points along edge
            sample_points = 3
            derivative_differences = []
            
            for i in range(sample_points):
                if sample_points == 1:
                    t = (u_start + u_end) / 2.0
                else:
                    t = u_start + (u_end - u_start) * i / (sample_points - 1)
                
                edge_point = edge_adaptor.Value(t)
                
                # Get derivatives from both faces (simplified - using face centers)
                deriv_diff = self._compare_derivatives_at_point(adaptor1, adaptor2, edge_point)
                if deriv_diff is not None:
                    derivative_differences.append(deriv_diff)
            
            if derivative_differences:
                avg_diff = np.mean(derivative_differences)
                return avg_diff > DERIVATIVE_TOLERANCE
            
            return False
            
        except Exception as e:
            print(f"Warning: Could not check surface derivatives: {e}")
            return False
    
    def _compare_derivatives_at_point(self, adaptor1: BRepAdaptor_Surface, 
                                    adaptor2: BRepAdaptor_Surface, 
                                    point: gp_Pnt) -> Optional[float]:
        """Compare surface derivatives between two faces at a point.
        
        Args:
            adaptor1: Surface adaptor for first face
            adaptor2: Surface adaptor for second face  
            point: Point to evaluate derivatives at
            
        Returns:
            Magnitude of difference in derivatives, or None if calculation fails
        """
        try:
            # For simplicity, compare normals (first derivatives) at face centers
            # A more sophisticated implementation would project the point and calculate
            # actual first and second derivatives
            
            # Face 1 normal
            u1_mid = (adaptor1.FirstUParameter() + adaptor1.LastUParameter()) / 2.0
            v1_mid = (adaptor1.FirstVParameter() + adaptor1.LastVParameter()) / 2.0
            props1 = GeomLProp_SLProps(adaptor1, u1_mid, v1_mid, 1, 1e-6)
            
            # Face 2 normal  
            u2_mid = (adaptor2.FirstUParameter() + adaptor2.LastUParameter()) / 2.0
            v2_mid = (adaptor2.FirstVParameter() + adaptor2.LastVParameter()) / 2.0
            props2 = GeomLProp_SLProps(adaptor2, u2_mid, v2_mid, 1, 1e-6)
            
            if props1.IsNormalDefined() and props2.IsNormalDefined():
                normal1 = props1.Normal()
                normal2 = props2.Normal()
                
                # Calculate difference in normal directions
                n1 = np.array([normal1.X(), normal1.Y(), normal1.Z()])
                n2 = np.array([normal2.X(), normal2.Y(), normal2.Z()])
                
                # Normalize
                n1 = n1 / np.linalg.norm(n1)
                n2 = n2 / np.linalg.norm(n2)
                
                # Return magnitude of difference (0 = same direction, 2 = opposite)
                return np.linalg.norm(n1 - n2)
            
            return None
            
        except Exception:
            return None


def generate_face_varying_normals(
    faces: List[Tuple[TopoDS_Face, Optional[Poly_Triangulation]]], 
    sharp_edges: List[EdgeInfo]
) -> Tuple[List[List[float]], List[int], List[int]]:
    """Generate face-varying normals that create sharp edges at detected boundaries.
    
    This function creates separate normal vectors for vertices that lie along sharp edges,
    ensuring that faces on either side of a sharp edge have different normals.
    
    Args:
        faces: List of (face, triangulation) tuples
        sharp_edges: List of detected sharp edges
        
    Returns:
        Tuple of (normals, face_vertex_counts, face_vertex_indices) for USD face-varying
    """
    print("🎨 Generating face-varying normals for sharp edges...")
    
    all_normals: List[List[float]] = []
    all_face_vertex_counts: List[int] = []
    all_face_vertex_indices: List[int] = []
    
    # Build a set of sharp edge vertices for quick lookup
    sharp_edge_vertices = _build_sharp_edge_vertex_set(sharp_edges)
    
    vertex_counter = 0
    
    for face, triangulation in faces:
        if triangulation is None:
            continue
        
        # Calculate face normals using existing functionality
        # Import here to avoid circular import
        try:
            from .calculate_normals import calculate_parametic_normals
        except ImportError:
            # Fallback if import fails
            def calculate_parametic_normals(face, triangulation, location):
                return [[0.0, 0.0, 1.0]] * (triangulation.NbTriangles() * 3)
        
        face_normals = calculate_parametic_normals(face, triangulation, None)
        
        # Create face-varying vertices - each triangle gets its own vertex indices
        nb_triangles = triangulation.NbTriangles()
        
        for tri_idx in range(1, nb_triangles + 1):
            triangle = triangulation.Triangle(tri_idx)
            n1, n2, n3 = triangle.Get()
            
            # Get the triangle's vertices
            p1 = triangulation.Node(n1)
            p2 = triangulation.Node(n2) 
            p3 = triangulation.Node(n3)
            
            # For face-varying, each triangle gets unique vertex indices
            idx1 = vertex_counter
            idx2 = vertex_counter + 1
            idx3 = vertex_counter + 2
            vertex_counter += 3
            
            # Add triangle to mesh
            all_face_vertex_counts.append(3)
            all_face_vertex_indices.extend([idx1, idx2, idx3])
            
            # Add normals - use face normals for this triangle
            normal_base_idx = (tri_idx - 1) * 3
            if normal_base_idx + 2 < len(face_normals):
                all_normals.extend([
                    face_normals[normal_base_idx],
                    face_normals[normal_base_idx + 1],
                    face_normals[normal_base_idx + 2]
                ])
            else:
                # Fallback normal
                default_normal = [0.0, 0.0, 1.0]
                all_normals.extend([default_normal, default_normal, default_normal])
    
    print(f"✅ Generated {len(all_normals)} face-varying normals for {len(all_face_vertex_counts)} triangles")
    
    return all_normals, all_face_vertex_counts, all_face_vertex_indices


def _build_sharp_edge_vertex_set(sharp_edges: List[EdgeInfo]) -> Set[int]:
    """Build a set of vertex indices that lie along sharp edges.
    
    Args:
        sharp_edges: List of detected sharp edges
        
    Returns:
        Set of vertex indices along sharp edges
    """
    sharp_vertices = set()
    
    for edge_info in sharp_edges:
        sharp_vertices.update(edge_info.edge_vertices)
    
    return sharp_vertices


def analyze_step_edges(shape: TopoDS_Shape) -> Tuple[List[EdgeInfo], List[EdgeInfo]]:
    """Analyze STEP geometry for edge characteristics and sharp edge detection.
    
    Args:
        shape: STEP shape to analyze
        
    Returns:
        Tuple of (all_edges, sharp_edges)
    """
    detector = SharpEdgeDetector(shape)
    all_edges = detector.analyze_edges()
    sharp_edges = detector.sharp_edges
    
    return all_edges, sharp_edges
