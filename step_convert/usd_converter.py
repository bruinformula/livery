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

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID
from pxr import UsdGeom, Gf

from .name_utils import sanitize_usd_name, generate_unique_name
from .geometry_processor import triangulate_shape, extract_faces, apply_location_to_shape
from .calculate_normals import calculate_face_normals, ensure_consistent_winding_order

try:
    from .config import FORCE_CONSISTENT_WINDING
except ImportError:
    # Default values if config not available
    FLIP_NORMALS = False
    FORCE_CONSISTENT_WINDING = True

def gp_trsf_to_usd_matrix(trsf):
    """Convert gp_Trsf to USD 4x4 matrix with proper precision."""
    # Get the transformation matrix components (OpenCASCADE uses 1-based indexing)
    # USD expects row-major 4x4 matrix
    translation = trsf.TranslationPart()
    
    # Build the 4x4 transformation matrix with proper float precision
    # Row 0: [m11, m12, m13, tx]
    # Row 1: [m21, m22, m23, ty] 
    # Row 2: [m31, m32, m33, tz]
    # Row 3: [0,   0,   0,   1]
    
    # Clean up very small values to avoid -0.0 display issues
    def clean_float(val):
        return 0.0 if abs(val) < 1e-10 else float(val)
    
    return Gf.Matrix4d(
        clean_float(trsf.Value(1, 1)), clean_float(trsf.Value(1, 2)), clean_float(trsf.Value(1, 3)), clean_float(translation.X()),
        clean_float(trsf.Value(2, 1)), clean_float(trsf.Value(2, 2)), clean_float(trsf.Value(2, 3)), clean_float(translation.Y()),
        clean_float(trsf.Value(3, 1)), clean_float(trsf.Value(3, 2)), clean_float(trsf.Value(3, 3)), clean_float(translation.Z()),
        0.0, 0.0, 0.0, 1.0
    )

def toploc_to_usd_matrix(location):
    """Convert TopLoc_Location to USD 4x4 matrix."""
    if location.IsIdentity():
        return Gf.Matrix4d(1.0)  # Identity matrix
    
    trsf = location.Transformation()
    return gp_trsf_to_usd_matrix(trsf)

def create_geometry_instances(stage, hierarchy_list, master_meshes):
    """Create USD geometry instances for duplicate shapes.
    
    Args:
        stage: USD stage
        hierarchy_list: List of shape hierarchy information  
        master_meshes: Dict mapping product_name to master mesh path
    """
    print(f"Processing geometry instances...")
    instance_count = 0
    
    def process_shape_instances(shape_info):
        nonlocal instance_count
        
        # Check if this is a leaf node with geometry
        if not shape_info['children'] and not shape_info.get('is_assembly', False):
            if 'usd_path' in shape_info:
                product_name = shape_info.get('product_name', shape_info['name'])
                
                if product_name in master_meshes:
                    # Create a reference to the master mesh geometry
                    usd_prim_path = shape_info['usd_path']
                    mesh_name = f"{sanitize_usd_name(product_name)}_mesh"
                    mesh_path = usd_prim_path.AppendChild(mesh_name)
                    
                    # Get the master mesh path
                    master_path = master_meshes[product_name]
                    
                    # Only create instance if this isn't the master
                    if str(mesh_path) != str(master_path):
                        # Create a reference/instance
                        mesh_prim = stage.DefinePrim(mesh_path)
                        mesh_prim.GetReferences().AddInternalReference(master_path)
                        instance_count += 1
                        print(f"    Created instance {instance_count}: {mesh_path} -> {master_path}")
        
        # Process children recursively
        if shape_info['children']:
            for child_info in shape_info['children']:
                process_shape_instances(child_info)
    
    for shape_info in hierarchy_list:
        process_shape_instances(shape_info)
    
    print(f"    Created {instance_count} geometry instances")

def create_usd_hierarchy_structure(stage, parent_prim, shape_info, depth=0, created_prims=None):
    """Create USD hierarchy structure with transformations preserved in USD space.
    
    Args:
        stage: USD stage
        parent_prim: Parent USD prim 
        shape_info: Dictionary containing shape data and hierarchy
        depth: Current depth in hierarchy for indentation
        created_prims: Set to track already created prims (to avoid duplicates)
        
    Returns:
        The created USD Xform prim
    """
    
    if created_prims is None:
        created_prims = set()
    
    indent = "  " * depth
    nauo_info = f" (NAUO: {shape_info.get('nauo_id', 'N/A')})" if 'nauo_id' in shape_info else ""
    product_info = f" (PRODUCT: {shape_info.get('product_name', 'N/A')})" if shape_info.get('product_name') else ""
    
    # Use PRODUCT name as the primary USD object name, fallback to regular name
    if shape_info.get('product_name'):
        usd_object_name = shape_info['product_name']
    else:
        usd_object_name = shape_info['name']
    
    sanitized_name = sanitize_usd_name(usd_object_name)
    if not sanitized_name or sanitized_name == "unnamed":
        sanitized_name = generate_unique_name()
    
    # Create a unique path based on the hierarchy position to avoid conflicts
    # Use the NAUO ID to make it unique if available
    if shape_info.get('nauo_id'):
        unique_name = f"{sanitized_name}_{shape_info['nauo_id']}"
    else:
        unique_name = sanitized_name
    
    # Create an Xform for this shape/assembly
    xform_path = parent_prim.GetPath().AppendChild(unique_name)
    
    # Check if this exact path was already created
    if str(xform_path) in created_prims:
        print(f"{indent}  Skipping duplicate prim: {xform_path}")
        return stage.GetPrimAtPath(xform_path)
    
    created_prims.add(str(xform_path))
    xform = UsdGeom.Xform.Define(stage, xform_path)
    
    # Store the USD path in the shape_info for later use
    shape_info['usd_path'] = xform_path
    
    # Apply the component's transformation to the USD Xform
    location = shape_info.get('location')
    if location and not location.IsIdentity():
        print(f"{indent}  Applying transformation to USD Xform: {unique_name}")
        usd_matrix = toploc_to_usd_matrix(location)
        
        # Apply the full transformation matrix to the USD Xform
        # Clear any existing transforms first (to prevent duplicates)
        xformable = UsdGeom.Xformable(xform)
        xformable.ClearXformOpOrder()
        
        # Add a single transform operation
        transform_attr = xformable.AddTransformOp()
        transform_attr.Set(usd_matrix)
        
        print(f"{indent}    Applied 4x4 matrix transformation")
        
    else:
        print(f"{indent}  Identity transformation for: {unique_name}")
    
    # Add metadata attributes to the USD prim
    prim = xform.GetPrim()
    if shape_info.get('nauo_id'):
        prim.SetCustomDataByKey('nauo_id', shape_info['nauo_id'])
    if shape_info.get('product_name'):
        prim.SetCustomDataByKey('product_name', shape_info['product_name'])
    
    # Set display name as USD metadata
    prim.SetDisplayName(shape_info['name'])
    
    # Process children recursively
    if shape_info['children']:
        for child_info in shape_info['children']:
            create_usd_hierarchy_structure(stage, xform, child_info, depth + 1, created_prims)
    
    return xform

def convert_hierarchical_shape_to_usd(stage, parent_prim, shape_info, shape_tool, depth=0, accumulated_transform=None):
    """Convert a hierarchical shape to USD, with proper transformation accumulation.
    
    Args:
        stage: USD stage
        parent_prim: Parent USD prim 
        shape_info: Dictionary containing shape data and hierarchy
        shape_tool: XCAF shape tool for additional operations
        depth: Current depth in hierarchy for indentation
        accumulated_transform: Accumulated transformation from parent hierarchy
    """
    
    indent = "  " * depth
    nauo_info = f" (NAUO: {shape_info.get('nauo_id', 'N/A')})" if 'nauo_id' in shape_info else ""
    product_info = f" (PRODUCT: {shape_info.get('product_name', 'N/A')})" if shape_info.get('product_name') else ""
    
    if shape_info.get('product_name'):
        usd_object_name = shape_info['product_name']
    else:
        usd_object_name = shape_info['name']
    
    if shape_info.get('nauo_id'):
        unique_name = f"{sanitize_usd_name(usd_object_name)}_{shape_info['nauo_id']}"
    else:
        unique_name = sanitize_usd_name(usd_object_name)
        if not unique_name or unique_name == "unnamed":
            unique_name = generate_unique_name()
    
    sanitized_name = unique_name
    
    xform_path = parent_prim.GetPath().AppendChild(sanitized_name)
    xform = UsdGeom.Xform.Define(stage, xform_path)
    
    current_location = shape_info.get('location')
    if accumulated_transform is None:
        from OCC.Core.TopLoc import TopLoc_Location
        accumulated_transform = TopLoc_Location()
    
    if current_location and not current_location.IsIdentity():
        total_transform = accumulated_transform * current_location
    else:
        total_transform = accumulated_transform
    
    prim = xform.GetPrim()
    if shape_info.get('nauo_id'):
        prim.SetCustomDataByKey('nauo_id', shape_info['nauo_id'])
    if shape_info.get('product_name'):
        prim.SetCustomDataByKey('product_name', shape_info['product_name'])
    
    prim.SetDisplayName(shape_info['name'])
    
    if not shape_info['children'] and not shape_info.get('is_assembly', False):
        shape = shape_info['shape']
        
        if total_transform and not total_transform.IsIdentity():
            shape = apply_location_to_shape(shape, total_transform)
        
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        if explorer.More():
            mesh_base_name = shape_info.get('product_name') or shape_info['name']
            mesh_name = f"{sanitize_usd_name(mesh_base_name)}_mesh"
            try:
                mesh_prim = convert_shape_to_usd_mesh(stage, xform, shape, mesh_name)
                if mesh_prim:
                    # Get mesh statistics for display
                    mesh_points = mesh_prim.GetPointsAttr().Get()
                    mesh_face_counts = mesh_prim.GetFaceVertexCountsAttr().Get()
                    vertex_count = len(mesh_points) if mesh_points else 0
                    face_count = len(mesh_face_counts) if mesh_face_counts else 0
                    print(f"{indent}Created: {mesh_name} with {vertex_count} vertices and {face_count} faces")
            except Exception as e:
                print(f"{indent}Warning: Failed to create mesh for {shape_info['name']}: {e}")
        else:
            print(f"{indent}Part {shape_info['name']} has no solid geometry (might be surface/wire)")
    else:
        print(f"{indent}Processing: {shape_info['name']}{nauo_info}{product_info}")
    
    if shape_info['children']:
        for i, child_info in enumerate(shape_info['children']):
            try:
                convert_hierarchical_shape_to_usd(stage, xform, child_info, shape_tool, depth + 1, total_transform)
            except Exception as e:
                print(f"{indent}  Warning: Failed to convert child {i} of {shape_info['name']}: {e}")
    
    return xform


def convert_shape_to_usd_mesh(stage, parent_prim, shape, name):
    """Convert a triangulated shape to a USD Mesh (geometry already in world space)."""
    
    triangulate_shape(shape)

    faces = extract_faces(shape)
    
    if not faces:
        print(f"    Warning: No triangulated faces found for shape {name}")
        return None
    
    all_points = []
    all_face_vertex_counts = []
    all_face_vertex_indices = []
    all_normals = []
    vertex_map = {}
    next_vertex_index = 0
    
    for face, triangulation in faces:
        if triangulation is None:
            continue
            
        face_normals = calculate_face_normals(face, triangulation, None)
        
        face_vertex_map = {}
        
        for i in range(1, triangulation.NbNodes() + 1):
            node = triangulation.Node(i)
            vertex = (float(node.X()), float(node.Y()), float(node.Z()))
            
            vertex_key = tuple(round(coord, 6) for coord in vertex)
            
            if vertex_key not in vertex_map:
                vertex_map[vertex_key] = next_vertex_index
                all_points.append(vertex)
                next_vertex_index += 1
            
            face_vertex_map[i] = vertex_map[vertex_key]
        
        normal_index = 0
        for i in range(1, triangulation.NbTriangles() + 1):
            triangle = triangulation.Triangle(i)
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
                        default_normal = [0, 0, 1]
                        all_normals.extend([default_normal, default_normal, default_normal])
                    
                    normal_index += 1
    
    if not all_points:
        print(f"    Warning: No vertices found for shape {name}")
        return None

    if FORCE_CONSISTENT_WINDING and all_normals:
        all_face_vertex_indices, all_normals = ensure_consistent_winding_order(
            all_points, all_face_vertex_indices, all_normals
        )
    
    mesh_path = parent_prim.GetPath().AppendChild(name)
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    
    mesh.CreatePointsAttr().Set(all_points)
    mesh.CreateFaceVertexCountsAttr().Set(all_face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr().Set(all_face_vertex_indices)
    
    if all_normals:
        mesh.CreateNormalsAttr().Set(all_normals)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    
    return mesh

