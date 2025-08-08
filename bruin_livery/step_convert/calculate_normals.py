from typing import List, Tuple, Dict, Optional, Any, Union
import numpy as np
from numpy.typing import NDArray
from OCC.Core.TopAbs import TopAbs_Orientation
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.Poly import Poly_Triangulation
from OCC.Core.TopLoc import TopLoc_Location

try:
    from .config import FLIP_NORMALS, FORCE_CONSISTENT_WINDING
except ImportError:
    # Default values if config not available
    FLIP_NORMALS: bool = False
    FORCE_CONSISTENT_WINDING: bool = True

def calculate_face_normals(
    face: TopoDS_Face, 
    triangulation: Poly_Triangulation, 
    location: Optional[TopLoc_Location] = None
) -> List[List[float]]:
    """Calculate proper vertex normals by accumulating face normals at each vertex. """
    normals: List[List[float]] = []
    
    try:
        # Access triangulation data using the correct API
        nb_nodes: int = triangulation.NbNodes()
        nb_triangles: int = triangulation.NbTriangles()
        
        if nb_nodes == 0 or nb_triangles == 0:
            return normals
        
        # Get surface normal directly from the face geometry if possible
        face_surface_normal: Optional[NDArray[np.float64]] = None
        try:
            
            # Try to get the face's surface normal direction
            adaptor: BRepAdaptor_Surface = BRepAdaptor_Surface(face)
            u_min: float = adaptor.FirstUParameter()
            u_max: float = adaptor.LastUParameter()
            v_min: float = adaptor.FirstVParameter()
            v_max: float = adaptor.LastVParameter()
            u_mid: float = (u_min + u_max) / 2
            v_mid: float = (v_min + v_max) / 2
            
            props: GeomLProp_SLProps = GeomLProp_SLProps(adaptor, u_mid, v_mid, 1, 1e-6)
            if props.IsNormalDefined():
                surface_normal = props.Normal()
                face_surface_normal = np.array([surface_normal.X(), surface_normal.Y(), surface_normal.Z()])
                
                # Apply face orientation
                if face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED:
                    face_surface_normal = -face_surface_normal
        except:
            face_surface_normal = None
        
        # Get all vertices and triangles
        vertex_normals: Dict[int, NDArray[np.float64]] = {}  # vertex_index -> accumulated normal
        vertex_counts: Dict[int, int] = {}   # vertex_index -> count of adjacent faces
        
        # First pass: calculate face normals and accumulate at vertices
        for i in range(1, nb_triangles + 1):
            triangle = triangulation.Triangle(i)
            n1: int
            n2: int
            n3: int
            n1, n2, n3 = triangle.Get()
            
            # Get the 3D points using the correct API
            p1 = triangulation.Node(n1)
            p2 = triangulation.Node(n2)
            p3 = triangulation.Node(n3)
            
            # Transform points if location is provided
            if location and not location.IsIdentity():
                trsf = location.Transformation()
                p1_t = p1.Transformed(trsf)
                p2_t = p2.Transformed(trsf)
                p3_t = p3.Transformed(trsf)
            else:
                p1_t, p2_t, p3_t = p1, p2, p3
            
            # Calculate face normal using cross product
            v1: NDArray[np.float64] = np.array([p1_t.X(), p1_t.Y(), p1_t.Z()])
            v2: NDArray[np.float64] = np.array([p2_t.X(), p2_t.Y(), p2_t.Z()])
            v3: NDArray[np.float64] = np.array([p3_t.X(), p3_t.Y(), p3_t.Z()])
            
            edge1: NDArray[np.float64] = v2 - v1
            edge2: NDArray[np.float64] = v3 - v1
            
            # Cross product (edge1 × edge2)
            face_normal: NDArray[np.float64] = np.cross(edge1, edge2)
            
            # Normalize face normal
            length: float = np.linalg.norm(face_normal)
            if length > 1e-10:
                face_normal = face_normal / length
            else:
                face_normal = np.array([0, 0, 1])
            
            # Ensure normal points outward by comparing with surface normal if available
            if face_surface_normal is not None:
                # Calculate dot product to check direction
                dot: float = np.dot(face_normal, face_surface_normal)
                
                # If dot product is negative, flip the normal
                if dot < 0:
                    face_normal = -face_normal
            else:
                # Fallback: Check face orientation from triangulation winding order
                try:
                    if face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED:
                        face_normal = -face_normal
                except:
                    pass  # Use face normal as-is if orientation check fails
            
            # Apply global normal flip if configured
            if FLIP_NORMALS:
                face_normal = -face_normal
            
            # Accumulate this face normal at each vertex
            for vertex_idx in [n1, n2, n3]:
                if vertex_idx not in vertex_normals:
                    vertex_normals[vertex_idx] = np.zeros(3, dtype=np.float64)
                    vertex_counts[vertex_idx] = 0
                
                vertex_normals[vertex_idx] += face_normal
                vertex_counts[vertex_idx] += 1
        
        # Second pass: normalize accumulated vertex normals
        for vertex_idx in vertex_normals:
            count: int = vertex_counts[vertex_idx]
            if count > 0:
                # Average the accumulated normals
                normal: NDArray[np.float64] = vertex_normals[vertex_idx] / count
                
                # Normalize the averaged normal
                length = np.linalg.norm(normal)
                if length > 1e-10:
                    normal = normal / length
                else:
                    normal = np.array([0, 0, 1])
                
                vertex_normals[vertex_idx] = normal
        
        # Third pass: assign vertex normals to triangles
        for i in range(1, nb_triangles + 1):
            triangle = triangulation.Triangle(i)
            n1, n2, n3 = triangle.Get()
            
            # Get normals for each vertex
            normal1: NDArray[np.float64] = vertex_normals.get(n1, np.array([0, 0, 1], dtype=np.float64))
            normal2: NDArray[np.float64] = vertex_normals.get(n2, np.array([0, 0, 1], dtype=np.float64))
            normal3: NDArray[np.float64] = vertex_normals.get(n3, np.array([0, 0, 1], dtype=np.float64))
            
            # Add vertex normals for this triangle (convert to list for compatibility)
            normals.extend([normal1.tolist(), normal2.tolist(), normal3.tolist()])
            
    except Exception as e:
        print(f"    Warning: Could not calculate normals for face: {e}")
        # Return empty normals on error
        return normals
    
    return normals

def calculate_parametic_normals(
    face: TopoDS_Face, 
    triangulation: Poly_Triangulation, 
    location: Optional[TopLoc_Location] = None
) -> List[List[float]]:
    """Calculate vertex normals using OpenCASCADE's parametric surface (UV) approach.
    
    This method uses the face's underlying parametric surface to calculate
    precise normals at each vertex position by finding the UV parameters
    and evaluating the surface normal at those coordinates.
    
    Args:
        face: The TopoDS_Face containing the surface
        triangulation: The triangulation data for the face
        location: Optional transformation location
        
    Returns:
        List of normals as [x, y, z] lists, three per triangle
    """
    normals: List[List[float]] = []
    
    try:
        # Access triangulation data
        nb_nodes: int = triangulation.NbNodes()
        nb_triangles: int = triangulation.NbTriangles()
        
        if nb_nodes == 0 or nb_triangles == 0:
            return normals
        
        # Get the surface adaptor for UV parameter evaluation
        adaptor: BRepAdaptor_Surface = BRepAdaptor_Surface(face)
        
        # Get UV parameter bounds
        u_min: float = adaptor.FirstUParameter()
        u_max: float = adaptor.LastUParameter()
        v_min: float = adaptor.FirstVParameter()
        v_max: float = adaptor.LastVParameter()
        
        # Check if the triangulation has UV nodes (2D parameters)
        has_uv: bool = triangulation.HasUVNodes()
        
        # Cache for calculated vertex normals to avoid recalculation
        vertex_normal_cache: Dict[int, NDArray[np.float64]] = {}
        
        # Process each triangle
        for i in range(1, nb_triangles + 1):
            triangle = triangulation.Triangle(i)
            n1: int
            n2: int  
            n3: int
            n1, n2, n3 = triangle.Get()
            
            triangle_normals: List[NDArray[np.float64]] = []
            
            # Calculate normal for each vertex of the triangle
            for vertex_idx in [n1, n2, n3]:
                if vertex_idx in vertex_normal_cache:
                    # Use cached normal
                    normal = vertex_normal_cache[vertex_idx]
                else:
                    # Calculate new normal
                    normal = np.array([0, 0, 1], dtype=np.float64)  # Default fallback
                    
                    try:
                        if has_uv:
                            # Use UV coordinates if available
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
                            
                            # Find UV parameters by projection (this is approximate)
                            # For better accuracy, you might want to use BRepClass_FaceClassifier
                            # or GeomAPI_ProjectPointOnSurf, but this is a simpler approach
                            u = u_min + (u_max - u_min) * 0.5  # Fallback to center
                            v = v_min + (v_max - v_min) * 0.5
                        
                        # Calculate surface normal at UV coordinates
                        # Get the underlying Geom_Surface from the GeomAdaptor_Surface
                        geom_adaptor_surface = adaptor.Surface()
                        geom_surface = geom_adaptor_surface.Surface()
                        props: GeomLProp_SLProps = GeomLProp_SLProps(geom_surface, u, v, 1, 1e-6)
                        
                        if props.IsNormalDefined():
                            surface_normal = props.Normal()
                            normal = np.array([
                                surface_normal.X(), 
                                surface_normal.Y(), 
                                surface_normal.Z()
                            ], dtype=np.float64)
                            
                            # Normalize the normal vector
                            length: float = np.linalg.norm(normal)
                            if length > 1e-10:
                                normal = normal / length
                            else:
                                normal = np.array([0, 0, 1], dtype=np.float64)
                            
                            # Apply face orientation
                            if face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED:
                                normal = -normal
                                
                        else:
                            # Fallback: use geometric calculation
                            # Get the 3D point and try to estimate normal from nearby points
                            vertex_3d = triangulation.Node(vertex_idx)
                            if location and not location.IsIdentity():
                                trsf = location.Transformation()
                                vertex_3d = vertex_3d.Transformed(trsf)
                            
                            # This is a very basic fallback - in practice you might want
                            # to use the face normal calculation as backup
                            normal = np.array([0, 0, 1], dtype=np.float64)
                            
                    except Exception as e:
                        # Fallback to default normal if UV calculation fails
                        print(f"    Warning: UV normal calculation failed for vertex {vertex_idx}: {e}")
                        normal = np.array([0, 0, 1], dtype=np.float64)
                    
                    # Apply global normal flip if configured
                    if FLIP_NORMALS:
                        normal = -normal
                    
                    # Cache the calculated normal
                    vertex_normal_cache[vertex_idx] = normal
                
                triangle_normals.append(normal)
            
            # Add the three vertex normals for this triangle
            for normal in triangle_normals:
                normals.append(normal.tolist())
                
    except Exception as e:
        print(f"    Warning: Could not calculate parametric normals for face: {e}")
        # Return empty normals on error
        return normals
    
    return normals
    

def ensure_consistent_winding_order(
    vertices: List[Tuple[float, float, float]], 
    face_indices: List[int], 
    normals: List[List[float]]
) -> Tuple[List[int], List[List[float]]]:
    """Ensure triangles have consistent winding order based on normal direction.
    
    Args:
        vertices: List of vertex positions [(x,y,z), ...]
        face_indices: List of triangle vertex indices [i1, i2, i3, ...]
        normals: List of normals corresponding to face_indices
        
    Returns:
        Tuple of (corrected_face_indices, corrected_normals)
    """
    if not FORCE_CONSISTENT_WINDING or len(face_indices) % 3 != 0:
        return face_indices, normals
    
    corrected_indices: List[int] = []
    corrected_normals: List[List[float]] = []
    
    for i in range(0, len(face_indices), 3):
        i1: int = face_indices[i]
        i2: int = face_indices[i+1]
        i3: int = face_indices[i+2]
        
        # Get triangle vertices
        if i1 < len(vertices) and i2 < len(vertices) and i3 < len(vertices):
            v1: Tuple[float, float, float] = vertices[i1]
            v2: Tuple[float, float, float] = vertices[i2]
            v3: Tuple[float, float, float] = vertices[i3]
            
            # Calculate face normal from geometry
            edge1: List[float] = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]]
            edge2: List[float] = [v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]]
            
            # Cross product (edge1 × edge2)
            geom_normal: List[float] = [
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0]
            ]
            
            # Normalize
            length: float = (geom_normal[0]**2 + geom_normal[1]**2 + geom_normal[2]**2)**0.5
            if length > 1e-10:
                geom_normal = [geom_normal[0]/length, geom_normal[1]/length, geom_normal[2]/length]
            
            # Get corresponding stored normal (average of three vertex normals)
            if i < len(normals):
                avg_normal: List[float] = [
                    (normals[i][0] + normals[i+1][0] + normals[i+2][0]) / 3,
                    (normals[i][1] + normals[i+1][1] + normals[i+2][1]) / 3,
                    (normals[i][2] + normals[i+1][2] + normals[i+2][2]) / 3
                ]
                
                # Check if winding order matches normal direction
                dot: float = (geom_normal[0] * avg_normal[0] + 
                             geom_normal[1] * avg_normal[1] + 
                             geom_normal[2] * avg_normal[2])
                
                if dot < 0:
                    # Flip winding order to match normal
                    corrected_indices.extend([i1, i3, i2])  # Swap i2 and i3
                    # Also store corrected normals
                    corrected_normals.extend([normals[i], normals[i+2], normals[i+1]])
                else:
                    # Keep original order
                    corrected_indices.extend([i1, i2, i3])
                    corrected_normals.extend([normals[i], normals[i+1], normals[i+2]])
            else:
                # No normals available, keep original
                corrected_indices.extend([i1, i2, i3])
        else:
            # Invalid indices, keep original
            corrected_indices.extend([i1, i2, i3])
    
    return corrected_indices, corrected_normals