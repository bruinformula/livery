import numpy as np
from OCC.Core.TopAbs import TopAbs_Orientation
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomLProp import GeomLProp_SLProps

try:
    from .config import FLIP_NORMALS, FORCE_CONSISTENT_WINDING
except ImportError:
    # Default values if config not available
    FLIP_NORMALS = False
    FORCE_CONSISTENT_WINDING = True

def calculate_face_normals(face, triangulation, location=None):
    """Calculate proper vertex normals by accumulating face normals at each vertex. """
    normals = []
    
    try:
        # Access triangulation data using the correct API
        nb_nodes = triangulation.NbNodes()
        nb_triangles = triangulation.NbTriangles()
        
        if nb_nodes == 0 or nb_triangles == 0:
            return normals
        
        # Get surface normal directly from the face geometry if possible
        face_surface_normal = None
        try:
            
            # Try to get the face's surface normal direction
            adaptor = BRepAdaptor_Surface(face)
            u_min, u_max, v_min, v_max = adaptor.FirstUParameter(), adaptor.LastUParameter(), adaptor.FirstVParameter(), adaptor.LastVParameter()
            u_mid, v_mid = (u_min + u_max) / 2, (v_min + v_max) / 2
            
            props = GeomLProp_SLProps(adaptor, u_mid, v_mid, 1, 1e-6)
            if props.IsNormalDefined():
                surface_normal = props.Normal()
                face_surface_normal = np.array([surface_normal.X(), surface_normal.Y(), surface_normal.Z()])
                
                # Apply face orientation
                if face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED:
                    face_surface_normal = -face_surface_normal
        except:
            face_surface_normal = None
        
        # Get all vertices and triangles
        vertex_normals = {}  # vertex_index -> accumulated normal
        vertex_counts = {}   # vertex_index -> count of adjacent faces
        
        # First pass: calculate face normals and accumulate at vertices
        for i in range(1, nb_triangles + 1):
            triangle = triangulation.Triangle(i)
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
            v1 = np.array([p1_t.X(), p1_t.Y(), p1_t.Z()])
            v2 = np.array([p2_t.X(), p2_t.Y(), p2_t.Z()])
            v3 = np.array([p3_t.X(), p3_t.Y(), p3_t.Z()])
            
            edge1 = v2 - v1
            edge2 = v3 - v1
            
            # Cross product (edge1 × edge2)
            face_normal = np.cross(edge1, edge2)
            
            # Normalize face normal
            length = np.linalg.norm(face_normal)
            if length > 1e-10:
                face_normal = face_normal / length
            else:
                face_normal = np.array([0, 0, 1])
            
            # Ensure normal points outward by comparing with surface normal if available
            if face_surface_normal is not None:
                # Calculate dot product to check direction
                dot = np.dot(face_normal, face_surface_normal)
                
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
                    vertex_normals[vertex_idx] = np.zeros(3)
                    vertex_counts[vertex_idx] = 0
                
                vertex_normals[vertex_idx] += face_normal
                vertex_counts[vertex_idx] += 1
        
        # Second pass: normalize accumulated vertex normals
        for vertex_idx in vertex_normals:
            count = vertex_counts[vertex_idx]
            if count > 0:
                # Average the accumulated normals
                normal = vertex_normals[vertex_idx] / count
                
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
            normal1 = vertex_normals.get(n1, np.array([0, 0, 1]))
            normal2 = vertex_normals.get(n2, np.array([0, 0, 1]))
            normal3 = vertex_normals.get(n3, np.array([0, 0, 1]))
            
            # Add vertex normals for this triangle (convert to list for compatibility)
            normals.extend([normal1.tolist(), normal2.tolist(), normal3.tolist()])
            
    except Exception as e:
        print(f"    Warning: Could not calculate normals for face: {e}")
        # Return empty normals on error
        return normals
    
    return normals

def ensure_consistent_winding_order(vertices, face_indices, normals):
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
    
    corrected_indices = []
    corrected_normals = []
    
    for i in range(0, len(face_indices), 3):
        i1, i2, i3 = face_indices[i], face_indices[i+1], face_indices[i+2]
        
        # Get triangle vertices
        if i1 < len(vertices) and i2 < len(vertices) and i3 < len(vertices):
            v1, v2, v3 = vertices[i1], vertices[i2], vertices[i3]
            
            # Calculate face normal from geometry
            edge1 = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]]
            edge2 = [v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]]
            
            # Cross product (edge1 × edge2)
            geom_normal = [
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0]
            ]
            
            # Normalize
            length = (geom_normal[0]**2 + geom_normal[1]**2 + geom_normal[2]**2)**0.5
            if length > 1e-10:
                geom_normal = [geom_normal[0]/length, geom_normal[1]/length, geom_normal[2]/length]
            
            # Get corresponding stored normal (average of three vertex normals)
            if i < len(normals):
                avg_normal = [
                    (normals[i][0] + normals[i+1][0] + normals[i+2][0]) / 3,
                    (normals[i][1] + normals[i+1][1] + normals[i+2][1]) / 3,
                    (normals[i][2] + normals[i+1][2] + normals[i+2][2]) / 3
                ]
                
                # Check if winding order matches normal direction
                dot = (geom_normal[0] * avg_normal[0] + 
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
