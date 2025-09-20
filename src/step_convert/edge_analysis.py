from __future__ import annotations
from typing import List, Tuple, Dict, Set, Optional, Any
import numpy as np
from numpy.typing import NDArray

from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge
from OCC.Core.TopExp import topexp
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_Orientation
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.Poly import Poly_Triangulation
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.TopAbs import TopAbs_Orientation
from .calculate_normals import calculate_parametic_normals


DIHEDRAL_ANGLE_THRESHOLD = 160.0  # degrees - mark as sharp if angle < 160°
NORMAL_ANGLE_THRESHOLD = 20.0     # degrees - mark as sharp if normals differ by > 20°


class EdgeInfo:
    """Information about an edge between two faces."""
    
    def __init__(self, edge: TopoDS_Edge, face1: TopoDS_Face, face2: TopoDS_Face):
        self.edge = edge
        self.face1 = face1
        self.face2 = face2
        self.angle = 180.0
        self.is_sharp = False
        self.derivatives_differ = False
        self.edge_vertices: List[int] = []
        
    def __repr__(self) -> str:
        return f"EdgeInfo(angle={self.angle:.1f}°, sharp={self.is_sharp})"


class SharpEdgeDetector:
    """Simple, reliable sharp edge detector with proper cylindrical surface handling."""
    
    def __init__(self, shape: TopoDS_Shape):
        self.shape = shape
        self.face_edge_map: Dict[TopoDS_Face, List[EdgeInfo]] = {}
        self.sharp_edges: List[EdgeInfo] = []
        self.edge_vertex_map: Dict[TopoDS_Edge, Set[int]] = {}
        
    def analyze_edges(self) -> List[EdgeInfo]:
        """Simple edge analysis that properly handles cylindrical surfaces."""
        print("🔍 Simple edge analysis...")
        
        edge_face_map = self._build_edge_face_map()
        all_edges: List[EdgeInfo] = []
        sharp_count = 0
        
        for edge, faces in edge_face_map.items():
            if len(faces) == 2:
                f1, f2 = faces[0], faces[1]

                # 🚫 ignore cylindrical seam edges
                if self._is_seam_edge(edge, f1, f2):
                    # print("    🚫 seam edge; skipping")
                    continue

                edge_info = EdgeInfo(edge, f1, f2)
                edge_info.angle = self._edge_dihedral_angle(edge, f1, f2)  # replaced method below
                edge_info.is_sharp = self._is_edge_sharp(edge_info, f1, f2)

                if edge_info.is_sharp:
                    self.sharp_edges.append(edge_info)
                    sharp_count += 1
                    type1 = self._get_simple_surface_type(f1)
                    type2 = self._get_simple_surface_type(f2)
                    print(f"  🔪 Sharp edge: {type1}↔{type2} angle={edge_info.angle:.1f}°")
                
                all_edges.append(edge_info)
        
        print(f"📊 Found {sharp_count} sharp edges out of {len(all_edges)} total")
        return all_edges

    def _is_edge_sharp(self, edge_info: EdgeInfo, face1: TopoDS_Face, face2: TopoDS_Face) -> bool:
        t1 = self._get_simple_surface_type(face1)
        t2 = self._get_simple_surface_type(face2)

        # ✅ Caps: cylinder ↔ plane are always sharp
        if (t1 == "Cylinder" and t2 == "Plane") or (t1 == "Plane" and t2 == "Cylinder"):
            # print("    🎯 cylinder–plane cap")
            return True

        # Generic criteria for everything else
        if edge_info.angle < DIHEDRAL_ANGLE_THRESHOLD:
            return True

        if t1 != t2:
            return True

        normal_diff = self._calculate_normal_angle_difference(edge_info.edge, face1, face2)
        return normal_diff > NORMAL_ANGLE_THRESHOLD


    def _is_seam_edge(self, edge: TopoDS_Edge, face1: TopoDS_Face, face2: TopoDS_Face) -> bool:
        """True if this is the parametric seam of a periodic surface (e.g., cylinder)."""
        try:
            # Seam appears as the SAME face on both sides + edge is closed on that face.
            return face1.IsSame(face2) and BRep_Tool.IsClosed(edge, face1)
        except Exception:
            return False

    def _get_simple_surface_type(self, face: TopoDS_Face) -> str:
        """Return 'Plane', 'Cylinder', or 'Other' (no guessing)."""
        try:
            adaptor = BRepAdaptor_Surface(face)
            st = adaptor.GetType()
            if st == GeomAbs_Plane:
                return "Plane"
            if st == GeomAbs_Cylinder:
                return "Cylinder"
            return "Other"
        except Exception:
            return "Unknown"
    
    def _simple_dihedral_angle(self, edge: TopoDS_Edge, face1: TopoDS_Face, face2: TopoDS_Face) -> float:
        """Simple dihedral angle calculation with cylinder handling."""
        try:
            type1 = self._get_simple_surface_type(face1)
            type2 = self._get_simple_surface_type(face2)
            
            if type1 == "Cylinder" and type2 == "Cylinder":
                # Cylinder-cylinder edges are often sharp in cutouts
                return 90.0  
            
            # Get normals at face centers (simple approach)
            normal1 = self._get_face_center_normal(face1)
            normal2 = self._get_face_center_normal(face2)
            
            if normal1 is None or normal2 is None:
                return 180.0
            
            # Calculate angle between normals
            dot_product = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
            angle_between_normals = np.degrees(np.arccos(abs(dot_product)))
            
            # Convert to dihedral angle
            dihedral_angle = 180.0 - angle_between_normals
            
            return max(0.0, min(180.0, dihedral_angle))
            
        except Exception:
            return 180.0
    

    def _edge_dihedral_angle(self, edge: TopoDS_Edge, face1: TopoDS_Face, face2: TopoDS_Face) -> float:
        """Dihedral based on normals sampled at the edge midpoint."""
        try:
            # 3D midpoint on the edge
            c = BRepAdaptor_Curve(edge)
            t = 0.5 * (c.FirstParameter() + c.LastParameter())
            p = c.Value(t)

            # Underlying surfaces
            s1 = BRep_Tool.Surface(face1)
            s2 = BRep_Tool.Surface(face2)

            # Project point to each surface to get (u,v)
            uv1 = GeomAPI_ProjectPointOnSurf(p, s1).LowerDistanceParameters()
            uv2 = GeomAPI_ProjectPointOnSurf(p, s2).LowerDistanceParameters()

            props1 = GeomLProp_SLProps(s1, uv1[0], uv1[1], 1, 1e-6)
            props2 = GeomLProp_SLProps(s2, uv2[0], uv2[1], 1, 1e-6)

            if not (props1.IsNormalDefined() and props2.IsNormalDefined()):
                return 180.0

            n1 = np.array([props1.Normal().X(), props1.Normal().Y(), props1.Normal().Z()], dtype=float)
            n2 = np.array([props2.Normal().X(), props2.Normal().Y(), props2.Normal().Z()], dtype=float)

            # Respect face orientation
            if face1.Orientation() == TopAbs_Orientation.TopAbs_REVERSED: n1 = -n1
            if face2.Orientation() == TopAbs_Orientation.TopAbs_REVERSED: n2 = -n2

            # Normalize
            n1 /= max(1e-15, np.linalg.norm(n1))
            n2 /= max(1e-15, np.linalg.norm(n2))

            dot = np.clip(np.dot(n1, n2), -1.0, 1.0)     # keep sign (do NOT abs)
            angle_between = np.degrees(np.arccos(dot))   # 0..180
            dihedral = 180.0 - angle_between             # 180=coplanar, 90=right angle

            return max(0.0, min(180.0, dihedral))
        except Exception:
            return 180.0

    def _calculate_normal_angle_difference(self, edge: TopoDS_Edge, face1: TopoDS_Face, face2: TopoDS_Face) -> float:
        """Calculate the angle difference between face normals, with special handling for cylinders."""
        try:
            type1 = self._get_simple_surface_type(face1)
            type2 = self._get_simple_surface_type(face2)
            
            # FIXED: For cylinder-cylinder interfaces, assume sharp (bypass faulty normal calculation)
            if type1 == "Cylinder" and type2 == "Cylinder":
                return 90.0  # Force detection as different normals
            
            normal1 = self._get_face_center_normal(face1)
            normal2 = self._get_face_center_normal(face2)
            
            if normal1 is None or normal2 is None:
                return 0.0
            
            dot_product = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
            angle = np.degrees(np.arccos(abs(dot_product)))
            
            return angle
            
        except Exception:
            return 0.0
    
    def _get_face_center_normal(self, face: TopoDS_Face) -> Optional[NDArray[np.float64]]:
        """Get normal at face center - simple and reliable."""
        try:
            adaptor = BRepAdaptor_Surface(face)
            
            u_center = (adaptor.FirstUParameter() + adaptor.LastUParameter()) / 2.0
            v_center = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2.0
            
            props = BRepLProp_SLProps(adaptor, u_center, v_center, 1, 1e-6)
            
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
            
            return None
            
        except Exception:
            return None
    
    def _build_edge_face_map(self) -> Dict[TopoDS_Edge, List[TopoDS_Face]]:
        """Build edge to face mapping."""
        edge_face_map: Dict[TopoDS_Edge, List[TopoDS_Face]] = {}
        
        map_tool = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndAncestors(self.shape, TopAbs_EDGE, TopAbs_FACE, map_tool)
        
        for i in range(1, map_tool.Size() + 1):
            edge = map_tool.FindKey(i)
            face_list = map_tool.FindFromIndex(i)
            
            faces = []
            if face_list.Size() > 0:
                faces.append(face_list.First())
                if face_list.Size() > 1:
                    faces.append(face_list.Last())
            
            edge_face_map[edge] = faces
        
        return edge_face_map

def _build_sharp_edge_vertex_set(sharp_edges: List[EdgeInfo]) -> Set[int]:
    """Build set of sharp edge vertices."""
    sharp_vertices = set()
    for edge_info in sharp_edges:
        sharp_vertices.update(edge_info.edge_vertices)
    return sharp_vertices


def analyze_step_edges(shape: TopoDS_Shape) -> Tuple[List[EdgeInfo], List[EdgeInfo]]:
    """Simple, reliable edge analysis.
    
    Args:
        shape: STEP shape to analyze
        
    Returns:
        Tuple of (all_edges, sharp_edges)
    """
    detector = SharpEdgeDetector(shape)
    all_edges = detector.analyze_edges()
    sharp_edges = detector.sharp_edges
    
    return all_edges, sharp_edges


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
    
    vertex_counter = 0
    
    for face, triangulation in faces:
        if triangulation is None:
            continue
        
        face_normals = calculate_parametic_normals(face, triangulation, None)
        
        # Flip normals if face orientation is reversed
        try:
            from OCC.Core.TopAbs import TopAbs_Orientation
            if face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED:
                face_normals = [[-n[0], -n[1], -n[2]] for n in face_normals]
        except Exception:
            pass

        nb_triangles = triangulation.NbTriangles()

        for tri_idx in range(1, nb_triangles + 1):
            triangle = triangulation.Triangle(tri_idx)

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
