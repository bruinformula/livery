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
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_Orientation
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
DERIVATIVE_DISCONTINUITY_THRESHOLD = 0.1  # degrees - minimal threshold for numerical precision only
DERIVATIVE_TOLERANCE = 1e-3  # tolerance for numerical precision in calculations


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
                
                # Check surface derivatives - this is the real test for sharp edges
                edge_info.derivatives_differ = self._check_surface_derivatives(edge, face1, face2)
                
                # Determine if edge is sharp based purely on derivative discontinuity
                edge_info.is_sharp = edge_info.derivatives_differ
                
                if edge_info.is_sharp:
                    self.sharp_edges.append(edge_info)
                    sharp_count += 1
                    print(f"  🔪 Sharp edge detected: derivative discontinuity at edge")
                
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
        
        This method properly handles curved surfaces by calculating normals at points
        near the actual edge rather than using face centers.
        
        Args:
            edge: The shared edge
            face1: First face
            face2: Second face
            
        Returns:
            Dihedral angle in degrees (0-180)
        """
        try:
            # Get edge curve
            edge_curve, first_param, last_param = BRep_Tool.Curve(edge)
            if edge_curve is None:
                return 180.0  # Assume flat if no curve data
                
            # Sample at the middle of the edge for most representative result
            mid_param = (first_param + last_param) / 2.0
            
            # Get point on edge
            edge_point = gp_Pnt()
            edge_curve.D0(mid_param, edge_point)
            
            # Get precise normals from both faces at this edge point
            normal1 = self._get_precise_surface_normal_at_edge(face1, edge_point)
            normal2 = self._get_precise_surface_normal_at_edge(face2, edge_point)
            
            if normal1 is None or normal2 is None:
                return 180.0  # Assume flat if normal calculation fails
            
            # Calculate dihedral angle between faces
            # The dihedral angle is the angle between the two face normals
            dot_product = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
            
            # Calculate angle between normals (0° to 180°)
            angle_rad = np.arccos(abs(dot_product))
            angle_deg = np.degrees(angle_rad)
            
            # For dihedral angles:
            # - Smooth surfaces: normals are nearly parallel, angle ≈ 0°, dihedral ≈ 180°
            # - Sharp edges: normals are at an angle, dihedral < 180°
            
            if dot_product > 0:
                # Normals point in similar direction (smooth transition)
                dihedral_angle = 180.0 - angle_deg
            else:
                # Normals point in different directions (potential discontinuity)
                dihedral_angle = 180.0 - angle_deg
            
            return max(0.0, min(180.0, dihedral_angle))  # Clamp to valid range
                
        except Exception as e:
            print(f"    Warning: Dihedral angle calculation failed: {e}")
            return 180.0  # Assume flat if calculation fails
    
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
        """Check if surface derivatives differ across an edge using proper geometric continuity.
        
        This method evaluates actual surface derivatives (tangent vectors) at the edge
        to determine if there's a true geometric discontinuity. For curved surfaces
        like cylinders, the normals should be continuous even though they change direction.
        
        Args:
            edge: The shared edge
            face1: First face  
            face2: Second face
            
        Returns:
            True if surface has derivative discontinuity (sharp edge)
        """
        try:
            # Get edge curve and sample points along it
            edge_curve, first_param, last_param = BRep_Tool.Curve(edge)
            if edge_curve is None:
                print("    ✅ Smooth edge: no curve data available")
                return False
            
            # Sample multiple points along the edge for robust analysis
            test_params = [
                first_param + 0.2 * (last_param - first_param),
                first_param + 0.5 * (last_param - first_param),
                first_param + 0.8 * (last_param - first_param)
            ]
            
            discontinuity_scores = []
            
            for param in test_params:
                try:
                    # Get point on edge
                    edge_point = gp_Pnt()
                    edge_curve.D0(param, edge_point)
                    
                    # Get surface derivatives from both faces at this edge point
                    derivatives1 = self._get_surface_derivatives_at_edge(face1, edge_point)
                    derivatives2 = self._get_surface_derivatives_at_edge(face2, edge_point)
                    
                    if derivatives1 is not None and derivatives2 is not None:
                        # Compare both first derivatives (tangent vectors)
                        normal1, du1, dv1 = derivatives1
                        normal2, du2, dv2 = derivatives2
                        
                        # For geometric continuity, check if the normals and tangent vectors align
                        # across the edge boundary. Small differences are expected for numerical precision.
                        
                        # Normal continuity check
                        normal_angle = self._angle_between_vectors(normal1, normal2)
                        
                        # Tangent continuity check - project tangents onto common plane
                        tangent_discontinuity = self._check_tangent_continuity(
                            du1, dv1, du2, dv2, normal1, normal2
                        )
                        
                        # Combine both checks for a continuity score
                        continuity_score = max(normal_angle, tangent_discontinuity)
                        discontinuity_scores.append(continuity_score)
                        
                except Exception as e:
                    print(f"    Warning: Derivative calculation failed at param {param}: {e}")
                    continue
            
            if not discontinuity_scores:
                print("    ✅ Smooth edge: could not analyze derivatives, assuming smooth")
                return False
            
            # Use the maximum discontinuity score across all sample points
            max_discontinuity = max(discontinuity_scores)
            
            # Pure derivative discontinuity detection - no angle threshold
            # If derivatives differ at all, it's a sharp edge
            DERIVATIVE_DISCONTINUITY_THRESHOLD = 0.1  # Very small threshold for numerical precision only
            
            is_discontinuous = max_discontinuity > DERIVATIVE_DISCONTINUITY_THRESHOLD
            
            return is_discontinuous
            
        except Exception as e:
            print(f"    Warning: Could not check derivatives: {e}")
            print("    ✅ Smooth edge: assuming smooth due to analysis failure")
            return False  # Assume smooth if analysis fails
    
    def _get_surface_derivatives_at_edge(self, face: TopoDS_Face, edge_point: gp_Pnt) -> Optional[Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]:
        """Get surface derivatives (normal and tangent vectors) at an edge point.
        
        Args:
            face: The face to analyze
            edge_point: Point on the edge
            
        Returns:
            Tuple of (normal, du_tangent, dv_tangent) vectors, or None if calculation fails
        """
        try:
            adaptor = BRepAdaptor_Surface(face)
            
            # Project the edge point onto the surface to get accurate UV coordinates
            from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
            
            # Get the underlying geometric surface
            geom_surface = BRep_Tool.Surface(face)
            if geom_surface is None:
                return None
            
            # Project point onto surface
            projector = GeomAPI_ProjectPointOnSurf(edge_point, geom_surface)
            
            if projector.NbPoints() > 0:
                # Get UV parameters of the closest point
                u, v = projector.Parameters(1)
                
                # Calculate surface derivatives (normal and tangent vectors)
                props = GeomLProp_SLProps(geom_surface, u, v, 2, 1e-6)  # 2nd order for curvature
                
                if props.IsNormalDefined():
                    # Get normal vector
                    normal = props.Normal()
                    normal_array = np.array([normal.X(), normal.Y(), normal.Z()])
                    
                    # Handle face orientation
                    if face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED:
                        normal_array = -normal_array
                    
                    # Normalize normal
                    normal_length = np.linalg.norm(normal_array)
                    if normal_length > 1e-10:
                        normal_array = normal_array / normal_length
                    else:
                        return None
                    
                    # Get first derivatives (tangent vectors)
                    du = gp_Vec()
                    dv = gp_Vec()
                    
                    try:
                        props.D1U(du)
                        props.D1V(dv)
                        
                        du_array = np.array([du.X(), du.Y(), du.Z()])
                        dv_array = np.array([dv.X(), dv.Y(), dv.Z()])
                        
                        # Normalize tangent vectors
                        du_length = np.linalg.norm(du_array)
                        dv_length = np.linalg.norm(dv_array)
                        
                        if du_length > 1e-10:
                            du_array = du_array / du_length
                        if dv_length > 1e-10:
                            dv_array = dv_array / dv_length
                        
                        return normal_array, du_array, dv_array
                    except Exception:
                        # If derivatives aren't available, return just the normal
                        return normal_array, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
            
            # Fallback: use face center if projection fails
            u_mid = (adaptor.FirstUParameter() + adaptor.LastUParameter()) / 2.0
            v_mid = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2.0
            
            props = BRepLProp_SLProps(adaptor, u_mid, v_mid, 2, 1e-6)
            if props.IsNormalDefined():
                normal = props.Normal()
                normal_array = np.array([normal.X(), normal.Y(), normal.Z()])
                
                # Handle face orientation
                if face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED:
                    normal_array = -normal_array
                
                normal_length = np.linalg.norm(normal_array)
                if normal_length > 1e-10:
                    normal_array = normal_array / normal_length
                    
                    # Get derivatives if available
                    du = gp_Vec()
                    dv = gp_Vec()
                    
                    try:
                        props.D1U(du)
                        props.D1V(dv)
                        
                        du_array = np.array([du.X(), du.Y(), du.Z()])
                        dv_array = np.array([dv.X(), dv.Y(), dv.Z()])
                        
                        du_length = np.linalg.norm(du_array)
                        dv_length = np.linalg.norm(dv_array)
                        
                        if du_length > 1e-10:
                            du_array = du_array / du_length
                        if dv_length > 1e-10:
                            dv_array = dv_array / dv_length
                        
                        return normal_array, du_array, dv_array
                    except Exception:
                        # If derivatives aren't available, return just the normal with default tangents
                        return normal_array, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
            
            return None
            
        except Exception as e:
            print(f"    Warning: Surface derivative calculation failed: {e}")
            return None
    
    def _angle_between_vectors(self, v1: NDArray[np.float64], v2: NDArray[np.float64]) -> float:
        """Calculate angle between two vectors in degrees.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Angle in degrees (0-180)
        """
        try:
            # Ensure vectors are normalized
            v1_norm = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 1e-10 else v1
            v2_norm = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 1e-10 else v2
            
            # Calculate dot product and clamp to valid range
            dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            
            # Calculate angle in radians then convert to degrees
            angle_rad = np.arccos(abs(dot_product))
            angle_deg = np.degrees(angle_rad)
            
            return angle_deg
            
        except Exception:
            return 0.0
    
    def _check_tangent_continuity(self, du1: NDArray[np.float64], dv1: NDArray[np.float64], 
                                du2: NDArray[np.float64], dv2: NDArray[np.float64],
                                normal1: NDArray[np.float64], normal2: NDArray[np.float64]) -> float:
        """Check continuity of tangent vectors across a surface boundary.
        
        For surfaces to be geometrically continuous (G1), the tangent vectors
        should align properly across the boundary.
        
        Args:
            du1, dv1: Tangent vectors for first surface
            du2, dv2: Tangent vectors for second surface  
            normal1, normal2: Normal vectors for both surfaces
            
        Returns:
            Discontinuity measure in degrees (0 = continuous)
        """
        try:
            # For curved surfaces like cylinders, the tangent spaces should align
            # We check if the tangent vectors from one surface can be expressed
            # as linear combinations of the tangent vectors from the other surface
            
            # Calculate the angle between tangent planes
            # The tangent plane is spanned by the du and dv vectors
            
            # Normal to tangent plane 1 (should be same as surface normal)
            plane_normal1 = np.cross(du1, dv1)
            plane_normal1_length = np.linalg.norm(plane_normal1)
            if plane_normal1_length > 1e-10:
                plane_normal1 = plane_normal1 / plane_normal1_length
            
            # Normal to tangent plane 2
            plane_normal2 = np.cross(du2, dv2)
            plane_normal2_length = np.linalg.norm(plane_normal2)
            if plane_normal2_length > 1e-10:
                plane_normal2 = plane_normal2 / plane_normal2_length
            
            # The angle between the tangent planes indicates continuity
            plane_angle = self._angle_between_vectors(plane_normal1, plane_normal2)
            
            # Also check individual tangent vector alignment
            # Project tangent vectors onto a common reference frame
            
            # Find the primary tangent directions
            du1_magnitude = np.linalg.norm(du1)
            dv1_magnitude = np.linalg.norm(dv1)
            du2_magnitude = np.linalg.norm(du2)
            dv2_magnitude = np.linalg.norm(dv2)
            
            # Use the dominant tangent direction for comparison
            primary_tangent1 = du1 if du1_magnitude > dv1_magnitude else dv1
            primary_tangent2 = du2 if du2_magnitude > dv2_magnitude else dv2
            
            tangent_angle = self._angle_between_vectors(primary_tangent1, primary_tangent2)
            
            # Return the maximum discontinuity
            return max(plane_angle, tangent_angle)
            
        except Exception:
            return 0.0  # Assume continuous if calculation fails

    def _get_precise_surface_normal_at_edge(self, face: TopoDS_Face, edge_point: gp_Pnt) -> Optional[NDArray[np.float64]]:
        """Get precise surface normal near an edge point using projection.
        
        This is a simplified version that extracts just the normal from the full derivative calculation.
        
        Args:
            face: The face to analyze
            edge_point: Point on the edge
            
        Returns:
            Unit normal vector as numpy array, or None if calculation fails
        """
        derivatives = self._get_surface_derivatives_at_edge(face, edge_point)
        if derivatives is not None:
            normal, _, _ = derivatives
            return normal
        return None
        """Get precise surface normal near an edge point using projection.
        
        Args:
            face: The face to analyze
            edge_point: Point on the edge
            
        Returns:
            Unit normal vector as numpy array, or None if calculation fails
        """
        try:
            adaptor = BRepAdaptor_Surface(face)
            
            # Project the edge point onto the surface to get accurate UV coordinates
            # This is more accurate than sampling a grid
            from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
            from OCC.Core.GeomAdaptor import GeomAdaptor_Surface
            
            # Get the underlying geometric surface
            geom_surface = BRep_Tool.Surface(face)
            if geom_surface is None:
                return None
            
            # Project point onto surface
            projector = GeomAPI_ProjectPointOnSurf(edge_point, geom_surface)
            
            if projector.NbPoints() > 0:
                # Get UV parameters of the closest point
                u, v = projector.Parameters(1)
                
                # Calculate surface normal at these UV coordinates
                props = GeomLProp_SLProps(geom_surface, u, v, 1, 1e-6)
                
                if props.IsNormalDefined():
                    normal = props.Normal()
                    normal_array = np.array([normal.X(), normal.Y(), normal.Z()])
                    
                    # Handle face orientation
                    if face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED:
                        normal_array = -normal_array
                    
                    # Normalize
                    length = np.linalg.norm(normal_array)
                    if length > 1e-10:
                        return normal_array / length
            
            # Fallback: use face center if projection fails
            u_mid = (adaptor.FirstUParameter() + adaptor.LastUParameter()) / 2.0
            v_mid = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2.0
            
            props = BRepLProp_SLProps(adaptor, u_mid, v_mid, 1, 1e-6)
            if props.IsNormalDefined():
                normal = props.Normal()
                normal_array = np.array([normal.X(), normal.Y(), normal.Z()])
                
                # Handle face orientation
                if face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED:
                    normal_array = -normal_array
                
                length = np.linalg.norm(normal_array)
                if length > 1e-10:
                    return normal_array / length
            
            return None
            
        except Exception:
            return None
    
    def _get_surface_normal_near_edge(self, face: TopoDS_Face, adaptor: BRepAdaptor_Surface, edge_point: gp_Pnt) -> Optional[NDArray[np.float64]]:
        """Get surface normal near an edge point by sampling the face surface.
        
        Args:
            face: The face to analyze
            adaptor: Surface adaptor for the face
            edge_point: Point on the edge
            
        Returns:
            Unit normal vector or None if calculation fails
        """
        try:
            # Sample UV parameters near the face center
            # For a proper implementation, we'd project the edge_point onto the surface
            # Here we use a simplified approach with small offsets from center
            u_min = adaptor.FirstUParameter()
            u_max = adaptor.LastUParameter()
            v_min = adaptor.FirstVParameter()
            v_max = adaptor.LastVParameter()
            
            u_center = (u_min + u_max) / 2.0
            v_center = (v_min + v_max) / 2.0
            
            # Try multiple sample points to find one that works
            test_points = [
                (u_center, v_center),                           # Center
                (u_center + (u_max - u_min) * 0.1, v_center),   # Slight offset
                (u_center, v_center + (v_max - v_min) * 0.1),   # Slight offset
                (u_center - (u_max - u_min) * 0.1, v_center),   # Opposite offset
                (u_center, v_center - (v_max - v_min) * 0.1),   # Opposite offset
            ]
            
            for u_test, v_test in test_points:
                try:
                    # Ensure UV parameters are within bounds
                    u_clamped = max(u_min, min(u_max, u_test))
                    v_clamped = max(v_min, min(v_max, v_test))
                    
                    # Calculate surface properties
                    props = BRepLProp_SLProps(adaptor, u_clamped, v_clamped, 1, 1e-6)
                    
                    if props.IsNormalDefined():
                        normal = props.Normal()
                        normal_array = np.array([normal.X(), normal.Y(), normal.Z()])
                        
                        # Normalize
                        length = np.linalg.norm(normal_array)
                        if length > 1e-10:
                            return normal_array / length
                
                except Exception:
                    continue
            
            return None
            
        except Exception:
            return None
    
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
        
        # Flip normals if face orientation is reversed
        try:
            from OCC.Core.TopAbs import TopAbs_Orientation
            #if face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED:
                #face_normals = [[-n[0], -n[1], -n[2]] for n in face_normals]
        except Exception:
            pass

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
    
    #print(f"✅ Generated {len(all_normals)} face-varying normals for {len(all_face_vertex_counts)} triangles")
    
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
