"""USD conversion and export functionality."""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID
from pxr import Usd, UsdGeom, Gf

from .name_utils import sanitize_usd_name, generate_unique_name
from .geometry_processor import triangulate_shape, extract_faces, apply_location_to_shape

class MeshData:
    """Container for mesh data that can be safely passed between threads."""
    def __init__(self, points, face_vertex_counts, face_vertex_indices, mesh_path, shape_info):
        self.points = points
        self.face_vertex_counts = face_vertex_counts
        self.face_vertex_indices = face_vertex_indices
        self.mesh_path = mesh_path
        self.shape_info = shape_info
        self.success = True
        self.error = None


class MeshTask:
    """Container for a mesh creation task."""
    def __init__(self, shape, mesh_path, shape_info, total_location):
        self.shape = shape
        self.mesh_path = mesh_path
        self.shape_info = shape_info
        self.total_location = total_location


def gp_trsf_to_usd_matrix(trsf):
    """Convert gp_Trsf to USD 4x4 matrix."""
    # Get the transformation matrix components (OpenCASCADE uses 1-based indexing)
    # USD expects row-major 4x4 matrix
    translation = trsf.TranslationPart()
    
    # Build the 4x4 transformation matrix
    # Row 0: [m11, m12, m13, tx]
    # Row 1: [m21, m22, m23, ty] 
    # Row 2: [m31, m32, m33, tz]
    # Row 3: [0,   0,   0,   1]
    return Gf.Matrix4d(
        trsf.Value(1, 1), trsf.Value(1, 2), trsf.Value(1, 3), translation.X(),
        trsf.Value(2, 1), trsf.Value(2, 2), trsf.Value(2, 3), translation.Y(),
        trsf.Value(3, 1), trsf.Value(3, 2), trsf.Value(3, 3), translation.Z(),
        0.0, 0.0, 0.0, 1.0
    )


def process_mesh_task(task):
    """Process a single mesh task in a thread-safe manner.
    
    Args:
        task: MeshTask containing shape and metadata
        
    Returns:
        MeshData: Processed mesh data or error information
    """
    try:
        shape = task.shape
        
        # Apply accumulated transformation to the shape before triangulation
        if not task.total_location.IsIdentity():
            shape = apply_location_to_shape(shape, task.total_location)
        
        # Triangulate the shape
        triangulate_shape(shape)
        
        # Extract faces
        faces = extract_faces(shape)
        
        if not faces:
            return MeshData([], [], [], task.mesh_path, task.shape_info)
        
        # Collect all vertices and face indices
        all_points = []
        all_face_vertex_counts = []
        all_face_vertex_indices = []
        vertex_offset = 0
        
        for face, triangulation in faces:
            if triangulation is None:
                continue
                
            # Get vertices from triangulation
            for i in range(1, triangulation.NbNodes() + 1):
                node = triangulation.Node(i)
                all_points.append((float(node.X()), float(node.Y()), float(node.Z())))
            
            # Get triangles from triangulation
            for i in range(1, triangulation.NbTriangles() + 1):
                triangle = triangulation.Triangle(i)
                n1, n2, n3 = triangle.Get()
                # Convert to 0-based indexing and add vertex offset
                all_face_vertex_counts.append(3)
                all_face_vertex_indices.extend([
                    n1 - 1 + vertex_offset,
                    n2 - 1 + vertex_offset, 
                    n3 - 1 + vertex_offset
                ])
            
            vertex_offset += triangulation.NbNodes()
        
        mesh_data = MeshData(all_points, all_face_vertex_counts, all_face_vertex_indices, task.mesh_path, task.shape_info)
        return mesh_data
        
    except Exception as e:
        mesh_data = MeshData([], [], [], task.mesh_path, task.shape_info)
        mesh_data.success = False
        mesh_data.error = str(e)
        return mesh_data



def create_mesh_prims_parallel(stage, mesh_data_list, max_workers=None):
    """Create USD mesh primitives from mesh data in parallel (thread-safe USD operations).
    
    Args:
        stage: USD stage
        mesh_data_list: List of MeshData objects
        max_workers: Maximum number of worker threads for USD operations
    """
    print(f"    🔄 Creating {len(mesh_data_list)} USD mesh primitives...")
    
    # USD operations need to be thread-safe, so we'll do them sequentially
    # but we can still process them efficiently
    successful_meshes = 0
    failed_meshes = 0
    
    for mesh_data in mesh_data_list:
        if not mesh_data.success:
            print(f"    ⚠️ Skipping failed mesh for {mesh_data.shape_info['name']}: {mesh_data.error}")
            failed_meshes += 1
            continue
            
        if not mesh_data.points:
            print(f"    ⚠️ Skipping mesh with no vertices for {mesh_data.shape_info['name']}")
            failed_meshes += 1
            continue
        
        try:
            # Check if the parent prim exists
            parent_path = mesh_data.mesh_path.GetParentPath()
            parent_prim = stage.GetPrimAtPath(parent_path)
            
            if not parent_prim.IsValid():
                print(f"    ❌ Parent prim does not exist at path: {parent_path}")
                print(f"    🔍 Creating missing parent prim as Xform...")
                parent_xform = UsdGeom.Xform.Define(stage, parent_path)
                if not parent_xform:
                    print(f"    ❌ Failed to create parent prim at path: {parent_path}")
                    failed_meshes += 1
                    continue
                
            # Create USD Mesh (this needs to be done on main thread for USD safety)
            print(f"    🔺 Creating mesh at path: {mesh_data.mesh_path}")
            mesh = UsdGeom.Mesh.Define(stage, mesh_data.mesh_path)
            
            if not mesh:
                print(f"    ❌ Failed to create USD mesh at path: {mesh_data.mesh_path}")
                failed_meshes += 1
                continue
            
            # Set mesh data
            mesh.CreatePointsAttr().Set(mesh_data.points)
            mesh.CreateFaceVertexCountsAttr().Set(mesh_data.face_vertex_counts)
            mesh.CreateFaceVertexIndicesAttr().Set(mesh_data.face_vertex_indices)
            
            successful_meshes += 1
            print(f"    ✅ Created mesh for {mesh_data.shape_info['name']} ({len(mesh_data.points)} vertices)")
            
        except Exception as e:
            print(f"    ⚠️ Failed to create USD mesh for {mesh_data.shape_info['name']}: {e}")
            print(f"    🔍 Attempted path: {mesh_data.mesh_path}")
            failed_meshes += 1
    
    print(f"    ✅ Created {successful_meshes} meshes successfully")
    if failed_meshes > 0:
        print(f"    ⚠️ Failed to create {failed_meshes} meshes")




def convert_hierarchical_shape_to_usd_parallel(stage, parent_prim, hierarchy_list, shape_tool, max_workers=None):
    """Convert hierarchical shapes to USD using parallel mesh processing.
    
    Args:
        stage: USD stage
        parent_prim: Parent USD prim
        hierarchy_list: List of shape hierarchy information
        shape_tool: XCAF shape tool
        max_workers: Maximum number of worker threads for mesh processing
    """
    
    print("🔄 Starting parallel mesh processing...")
    
    # First, create the USD hierarchy structure (must be done sequentially)
    print("📦 Creating USD hierarchy structure...")
    for i, shape_info in enumerate(hierarchy_list):
        print(f"🏗️ Processing structure for assembly {i+1}/{len(hierarchy_list)}: {shape_info['name']}")
        create_usd_hierarchy_structure(stage, parent_prim, shape_info)
    
    # Now collect mesh tasks using the stored USD paths
    print("📋 Collecting mesh creation tasks from hierarchy...")
    all_mesh_tasks = []
    for shape_info in hierarchy_list:
        tasks = collect_mesh_tasks_from_hierarchy(shape_info)
        all_mesh_tasks.extend(tasks)
    
    print(f"📊 Found {len(all_mesh_tasks)} mesh creation tasks")
    
    if not all_mesh_tasks:
        print("ℹ️ No meshes to create")
        return
    
    # Process mesh tasks in parallel
    print(f"⚡ Processing meshes in parallel (max_workers={max_workers})...")
    mesh_data_list = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all mesh processing tasks
        future_to_task = {executor.submit(process_mesh_task, task): task for task in all_mesh_tasks}
        
        # Collect results as they complete
        for i, future in enumerate(as_completed(future_to_task), 1):
            task = future_to_task[future]
            try:
                mesh_data = future.result()
                mesh_data_list.append(mesh_data)
                
                if mesh_data.success:
                    print(f"    ✅ ({i}/{len(all_mesh_tasks)}) Processed mesh for: {mesh_data.shape_info['name']}")
                else:
                    print(f"    ⚠️ ({i}/{len(all_mesh_tasks)}) Failed mesh for: {mesh_data.shape_info['name']}: {mesh_data.error}")
                    
            except Exception as e:
                print(f"    ❌ ({i}/{len(all_mesh_tasks)}) Exception processing mesh for: {task.shape_info['name']}: {e}")
    
    # Create USD mesh primitives (must be done sequentially for USD thread safety)
    print("🏗️ Creating USD mesh primitives...")
    create_mesh_prims_parallel(stage, mesh_data_list, max_workers)
    
    print("✅ Parallel mesh processing complete!")


def create_usd_hierarchy_structure(stage, parent_prim, shape_info, depth=0, accumulated_location=None):
    """Create USD hierarchy structure without mesh geometry (for parallel processing).
    
    Args:
        stage: USD stage
        parent_prim: Parent USD prim 
        shape_info: Dictionary containing shape data and hierarchy
        depth: Current depth in hierarchy for indentation
        accumulated_location: Accumulated transformation from root to this component
        
    Returns:
        The created USD Xform prim
    """
    
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
    
    # Create an Xform for this shape/assembly
    xform_path = parent_prim.GetPath().AppendChild(sanitized_name)
    xform = UsdGeom.Xform.Define(stage, xform_path)
    
    # Store the USD path in the shape_info for later use
    shape_info['usd_path'] = xform_path
    
    # Add metadata attributes to the USD prim
    prim = xform.GetPrim()
    if shape_info.get('nauo_id'):
        prim.SetCustomDataByKey('nauo_id', shape_info['nauo_id'])
    if shape_info.get('product_name'):
        prim.SetCustomDataByKey('product_name', shape_info['product_name'])
    if shape_info.get('component_entry'):
        prim.SetCustomDataByKey('component_entry', shape_info['component_entry'])
    if shape_info.get('referred_entry'):
        prim.SetCustomDataByKey('referred_entry', shape_info['referred_entry'])
    
    # Set display name as USD metadata
    prim.SetDisplayName(shape_info['name'])
    
    # Process children recursively
    if shape_info['children']:
        for child_info in shape_info['children']:
            create_usd_hierarchy_structure(stage, xform, child_info, depth + 1, accumulated_location)
    
    return xform


def collect_mesh_tasks_from_hierarchy(shape_info, accumulated_location=None, tasks=None):
    """Recursively collect mesh creation tasks from shape hierarchy using stored USD paths.
    
    Args:
        shape_info: Shape hierarchy information with stored USD paths
        accumulated_location: Accumulated transformation
        tasks: List to collect tasks in
        
    Returns:
        List of MeshTask objects
    """
    if tasks is None:
        tasks = []
    
    # Check if USD path was stored during hierarchy creation
    if 'usd_path' not in shape_info:
        print(f"⚠️ Warning: No USD path stored for shape {shape_info['name']}")
        return tasks
    
    usd_prim_path = shape_info['usd_path']
    
    # Accumulate transformations
    current_location = shape_info['location']
    if accumulated_location is None:
        total_location = current_location
    else:
        if not current_location.IsIdentity():
            total_location = accumulated_location.Multiplied(current_location)
        else:
            total_location = accumulated_location
    
    # Check if this is a leaf node with geometry
    if not shape_info['children'] and not shape_info.get('is_assembly', False):
        shape = shape_info['shape']
        
        # Check for solid geometry
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        if explorer.More():
            # Create mesh task with the stored USD prim path
            mesh_base_name = shape_info.get('product_name') or shape_info['name']
            mesh_name = f"{sanitize_usd_name(mesh_base_name)}_mesh"
            mesh_path = usd_prim_path.AppendChild(mesh_name)
            
            task = MeshTask(shape, mesh_path, shape_info, total_location)
            tasks.append(task)
    
    # Process children recursively
    if shape_info['children']:
        for child_info in shape_info['children']:
            collect_mesh_tasks_from_hierarchy(child_info, total_location, tasks)
    
    return tasks


def convert_hierarchical_shape_to_usd(stage, parent_prim, shape_info, shape_tool, depth=0, accumulated_location=None):
    """Convert a hierarchical shape to USD, preserving the hierarchy.
    
    Args:
        stage: USD stage
        parent_prim: Parent USD prim 
        shape_info: Dictionary containing shape data and hierarchy
        shape_tool: XCAF shape tool for additional operations
        depth: Current depth in hierarchy for indentation
        accumulated_location: Accumulated transformation from root to this component
    """
    
    indent = "  " * depth
    nauo_info = f" (NAUO: {shape_info.get('nauo_id', 'N/A')})" if 'nauo_id' in shape_info else ""
    product_info = f" (PRODUCT: {shape_info.get('product_name', 'N/A')})" if shape_info.get('product_name') else ""
    print(f"{indent}🔄 Converting: {shape_info['name']}{nauo_info}{product_info}")
    
    # Use PRODUCT name as the primary USD object name, fallback to regular name
    if shape_info.get('product_name'):
        usd_object_name = shape_info['product_name']
        print(f"{indent}  🏭 Using PRODUCT name for USD object: {usd_object_name}")
    else:
        usd_object_name = shape_info['name']
        print(f"{indent}  📝 Using regular name for USD object: {usd_object_name}")
    
    sanitized_name = sanitize_usd_name(usd_object_name)
    if not sanitized_name or sanitized_name == "unnamed":
        sanitized_name = generate_unique_name()
    
    # Create an Xform for this shape/assembly
    xform_path = parent_prim.GetPath().AppendChild(sanitized_name)
    xform = UsdGeom.Xform.Define(stage, xform_path)
    print(f"{indent}  📦 Created USD Xform: {xform_path}")
    
    # Add metadata attributes to the USD prim
    prim = xform.GetPrim()
    if shape_info.get('nauo_id'):
        prim.SetCustomDataByKey('nauo_id', shape_info['nauo_id'])
    if shape_info.get('product_name'):
        prim.SetCustomDataByKey('product_name', shape_info['product_name'])
    if shape_info.get('component_entry'):
        prim.SetCustomDataByKey('component_entry', shape_info['component_entry'])
    if shape_info.get('referred_entry'):
        prim.SetCustomDataByKey('referred_entry', shape_info['referred_entry'])
    
    # Set display name as USD metadata
    prim.SetDisplayName(shape_info['name'])
    
    # Accumulate transformations down the hierarchy
    current_location = shape_info['location']
    if accumulated_location is None:
        # This is the root, start with current location
        total_location = current_location
    else:
        # Combine parent transformation with current transformation
        if not current_location.IsIdentity():
            total_location = accumulated_location.Multiplied(current_location)
        else:
            total_location = accumulated_location
    
    # Only create mesh geometry for leaf components (parts with no children)
    # This prevents duplicate geometry for assemblies
    if not shape_info['children'] and not shape_info.get('is_assembly', False):
        shape = shape_info['shape']
        
        # Apply accumulated transformation to the shape before triangulation
        if not total_location.IsIdentity():
            print(f"{indent}  🔄 Applying accumulated transformation to geometry for: {shape_info['name']}")
            shape = apply_location_to_shape(shape, total_location)
        
        # Check for solid geometry
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        if explorer.More():
            # Use PRODUCT name for mesh if available
            mesh_base_name = shape_info.get('product_name') or shape_info['name']
            mesh_name = f"{sanitize_usd_name(mesh_base_name)}_mesh"
            print(f"{indent}  🔺 Creating mesh geometry for part...")
            try:
                mesh_prim = convert_shape_to_usd_mesh(stage, xform, shape, mesh_name)
                if mesh_prim:
                    print(f"{indent}  ✅ Created mesh for: {shape_info['name']}")
            except Exception as e:
                print(f"{indent}  ⚠️ Warning: Failed to create mesh for {shape_info['name']}: {e}")
        else:
            print(f"{indent}  ℹ️ Part {shape_info['name']} has no solid geometry (might be surface/wire)")
    elif shape_info['children']:
        print(f"{indent}  🏭 Assembly {shape_info['name']} - processing children only")
    else:
        print(f"{indent}  ℹ️ Empty assembly: {shape_info['name']}")
    
    # Process children recursively, passing down the accumulated transformation
    if shape_info['children']:
        print(f"{indent}  📁 Processing {len(shape_info['children'])} children...")
        for i, child_info in enumerate(shape_info['children']):
            try:
                convert_hierarchical_shape_to_usd(stage, xform, child_info, shape_tool, depth + 1, total_location)
            except Exception as e:
                print(f"{indent}  ⚠️ Warning: Failed to convert child {i} of {shape_info['name']}: {e}")
    
    print(f"{indent}✅ Completed: {shape_info['name']}")
    return xform


def convert_shape_to_usd_mesh(stage, parent_prim, shape, name):
    """Convert a triangulated shape to a USD Mesh."""
    
    print(f"    🔺 Triangulating shape...")
    # Shape should already have transformations applied by this point
    triangulate_shape(shape)

    # Gather all triangles from shape
    faces = extract_faces(shape)
    
    if not faces:
        print(f"    ⚠️ Warning: No triangulated faces found for shape {name}")
        return None
    
    print(f"    📐 Processing {len(faces)} faces...")
    # Collect all vertices and face indices
    all_points = []
    all_face_vertex_counts = []
    all_face_vertex_indices = []
    vertex_offset = 0
    
    for face, triangulation in faces:
        if triangulation is None:
            continue
            
        # Get vertices from triangulation
        for i in range(1, triangulation.NbNodes() + 1):
            node = triangulation.Node(i)
            all_points.append((float(node.X()), float(node.Y()), float(node.Z())))
        
        # Get triangles from triangulation
        for i in range(1, triangulation.NbTriangles() + 1):
            triangle = triangulation.Triangle(i)
            n1, n2, n3 = triangle.Get()
            # Convert to 0-based indexing and add vertex offset
            all_face_vertex_counts.append(3)
            all_face_vertex_indices.extend([
                n1 - 1 + vertex_offset,
                n2 - 1 + vertex_offset, 
                n3 - 1 + vertex_offset
            ])
        
        vertex_offset += triangulation.NbNodes()
    
    if not all_points:
        print(f"    ⚠️ Warning: No vertices found for shape {name}")
        return None
    
    print(f"    🏗️ Creating USD mesh with {len(all_points)} vertices, {len(all_face_vertex_counts)} triangles...")
    # Create USD Mesh
    mesh_path = parent_prim.GetPath().AppendChild(name)
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    
    # Set mesh data
    mesh.CreatePointsAttr().Set(all_points)
    mesh.CreateFaceVertexCountsAttr().Set(all_face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr().Set(all_face_vertex_indices)
    
    print(f"    ✅ Created mesh {name} with {len(all_points)} vertices and {len(all_face_vertex_counts)} faces")
    return mesh

