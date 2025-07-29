"""STEP file reading and hierarchy parsing functionality."""

from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.XCAFDoc import (XCAFDoc_DocumentTool_ShapeTool, 
                              XCAFDoc_DocumentTool_ColorTool,
                              XCAFDoc_DocumentTool)
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.XCAFApp import XCAFApp_Application
from OCC.Core.TDataStd import TDataStd_Name
from OCC.Core.TDF import TDF_LabelSequence, TDF_Tool, TDF_AttributeIterator
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TCollection import TCollection_AsciiString
from OCC.Extend.DataExchange import read_step_file


def read_step_file_with_hierarchy(step_file_path, step_product_names=None):
    """Read STEP file and return hierarchical structure with transformations."""
    if step_product_names is None:
        step_product_names = {}
        
    print("📖 Initializing XCAF document...")
    # Create a new document
    app = XCAFApp_Application.GetApplication()
    doc = TDocStd_Document("XCAF")
    app.NewDocument("STEP", doc)
    
    # Get the shape tool
    print("🔧 Setting up XCAF tools...")
    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool(doc.Main())
    
    # Read the STEP file
    print(f"📁 Reading STEP file: {step_file_path}")
    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetLayerMode(True)
    reader.SetNameMode(True)
    reader.SetMatMode(True)
    
    print("🔄 Processing STEP file...")
    status = reader.ReadFile(step_file_path)
    if status != 1:  # IFSelect_RetDone
        raise ValueError(f"❌ Error reading STEP file: {step_file_path}")
    
    print("🔄 Transferring data to XCAF document...")
    reader.Transfer(doc)
    
    # Get shapes at root level
    print("🔍 Analyzing hierarchical structure...")
    labels = TDF_LabelSequence()
    shape_tool.GetShapes(labels)
    
    if labels.Length() == 0:
        raise ValueError("No shapes found in STEP file")
    
    print(f"📊 Found {labels.Length()} labels at root")
    
    # Get the first label at root (this should be the top assembly)
    root_label = labels.Value(1)
    root_name = root_label.GetLabelName()
    if not root_name:
        root_name = "Top"
    
    print(f"🏗️ Root assembly: {root_name}")
    
    # Build hierarchy starting from root
    hierarchy = []
    if shape_tool.IsAssembly(root_label):
        print(f"    📦 Root is an assembly")
        root_shape = shape_tool.GetShape(root_label)
        
        root_info = {
            'label': root_label,
            'shape': root_shape,
            'name': root_name,
            'location': TopLoc_Location(),  # Root has no transformation
            'children': [],
            'is_assembly': True
        }
        
        # Get components of root assembly
        components = TDF_LabelSequence()
        shape_tool.GetComponents(root_label, components)
        print(f"    🔄 Root assembly has {components.Length()} components")
        
        # Parse components recursively
        _parse_components_recursive(shape_tool, color_tool, components, root_info['children'], step_product_names)
        
        hierarchy.append(root_info)
    else:
        print(f"    🔧 Root is a simple shape")
        # Handle case where root is just a shape, not an assembly
        root_shape = shape_tool.GetShape(root_label)
        root_info = {
            'label': root_label,
            'shape': root_shape,
            'name': root_name,
            'location': TopLoc_Location(),
            'children': [],
            'is_assembly': False
        }
        hierarchy.append(root_info)
    
    print(f"✅ Hierarchy analysis complete!")
    return hierarchy, shape_tool, color_tool


def _extract_product_name_from_label(shape_tool, label, default_name, step_product_names=None):
    """Extract PRODUCT name from STEP label attributes using multiple strategies."""
    if step_product_names is None:
        step_product_names = {}
        
    try:
        # Strategy 0: Check if we have a direct match in our extracted PRODUCT names
        # Try to match based on label entry or name
        label_entry_str = TCollection_AsciiString()
        TDF_Tool.Entry(label, label_entry_str)
        label_entry = label_entry_str.ToCString()
        
        # Look for PRODUCT name matches by entity ID or name
        for entity_id, product_info in step_product_names.items():
            product_name = product_info['name']
            # Check if the default name matches or is similar to the PRODUCT name
            if (default_name and 
                (default_name.lower() == product_name.lower() or
                 product_name.lower() in default_name.lower() or
                 default_name.lower() in product_name.lower())):
                print(f"        🎯 Matched PRODUCT by name: {product_name}")
                return product_name
            
            # Check if label entry contains the entity ID
            if entity_id in label_entry:
                print(f"        🎯 Matched PRODUCT by entity ID #{entity_id}: {product_name}")
                return product_name
        
        # Strategy 1: Look for TDataStd_Name attributes with more thorough search
        found_names = []
        
        # Check the label itself first
        try:
            # Try using Handle_TDF_Attribute approach
            from OCC.Core.TDF import Handle_TDF_Attribute
            attr_handle = Handle_TDF_Attribute()
            if label.FindAttribute(TDataStd_Name.GetID(), attr_handle):
                if not attr_handle.IsNull():
                    name_attr = TDataStd_Name.DownCast(attr_handle)
                    if name_attr and not name_attr.IsNull():
                        # Fixed: Use ToCString() directly, not Get().ToCString()
                        name_string = name_attr.ToCString()
                        if name_string and name_string != default_name:
                            found_names.append(name_string)
        except (ImportError, AttributeError, TypeError) as e:
            print(f"        ⚠️ Warning: Error extracting PRODUCT name: {e}")
            # Fallback: try alternative approaches
            try:
                # Try direct attribute access without FindAttribute
                name = label.GetLabelName()
                if name and name != default_name:
                    found_names.append(name)
            except:
                # Final fallback: skip this strategy
                pass
        
        # Strategy 2: Iterate through all attributes to find any name-like data
        attr_iter = TDF_AttributeIterator()
        attr_iter.Initialize(label, True)  # True to explore children
        
        while attr_iter.More():
            attr = attr_iter.Value()
            
            # Check if this is a name attribute
            if attr.DynamicType().Name() == 'TDataStd_Name':
                name_attr = TDataStd_Name.DownCast(attr)
                if name_attr:
                    # Fixed: Use ToCString() directly, not Get().ToCString()
                    name_string = name_attr.ToCString()
                    if name_string and name_string != default_name and name_string not in found_names:
                        found_names.append(name_string)
            
            attr_iter.Next()
        
        # Strategy 3: Try to get names from parent/child relationships
        # Sometimes PRODUCT names are stored at different levels
        parent_label = label.Father()
        if not parent_label.IsNull():
            try:
                from OCC.Core.TDF import Handle_TDF_Attribute
                attr_handle = Handle_TDF_Attribute()
                if parent_label.FindAttribute(TDataStd_Name.GetID(), attr_handle):
                    if not attr_handle.IsNull():
                        name_attr = TDataStd_Name.DownCast(attr_handle)
                        if name_attr and not name_attr.IsNull():
                            # Fixed: Use ToCString() directly, not Get().ToCString()
                            parent_name_string = name_attr.ToCString()
                            if parent_name_string and parent_name_string != default_name and parent_name_string not in found_names:
                                found_names.append(parent_name_string)
            except (ImportError, AttributeError, TypeError):
                # Fallback: try GetLabelName
                try:
                    parent_name = parent_label.GetLabelName()
                    if parent_name and parent_name != default_name and parent_name not in found_names:
                        found_names.append(parent_name)
                except:
                    pass
        
        # Strategy 4: Check child labels for names
        for i in range(1, label.NbChildren() + 1):
            child_label = label.FindChild(i)
            if not child_label.IsNull():
                try:
                    from OCC.Core.TDF import Handle_TDF_Attribute
                    attr_handle = Handle_TDF_Attribute()
                    if child_label.FindAttribute(TDataStd_Name.GetID(), attr_handle):
                        if not attr_handle.IsNull():
                            name_attr = TDataStd_Name.DownCast(attr_handle)
                            if name_attr and not name_attr.IsNull():
                                # Fixed: Use ToCString() directly, not Get().ToCString()
                                child_name_string = name_attr.ToCString()
                                if child_name_string and child_name_string != default_name and child_name_string not in found_names:
                                    found_names.append(child_name_string)
                except (ImportError, AttributeError, TypeError):
                    # Fallback: try GetLabelName
                    try:
                        child_name = child_label.GetLabelName()
                        if child_name and child_name != default_name and child_name not in found_names:
                            found_names.append(child_name)
                    except:
                        pass
        
        # Select the best candidate
        if found_names:
            # Prefer longer, more descriptive names that look like PRODUCT names
            for name in found_names:
                if (len(name) > 10 and  # Longer names are often PRODUCT names
                    ('_' in name or '-' in name) and  # PRODUCT names often have separators
                    not name.startswith('Component_') and  # Skip our generated names
                    not name.startswith('Shape_')):  # Skip generic shape names
                    print(f"        🔍 Found PRODUCT-like name: {name}")
                    return name
            
            # If no obvious PRODUCT name, return the longest meaningful name
            longest_name = max(found_names, key=len)
            if len(longest_name) > len(default_name or ""):
                print(f"        🔍 Using longest found name: {longest_name}")
                return longest_name
        
        # Strategy 5: Check if the default name itself looks like a PRODUCT name
        if (default_name and 
            len(default_name) > 5 and
            ('_' in default_name or '-' in default_name) and
            not default_name.startswith('Component_')):
            print(f"        🔍 Default name looks like PRODUCT: {default_name}")
            return default_name
            
    except Exception as e:
        print(f"        ⚠️ Warning: Error extracting PRODUCT name: {e}")
    
    return None


def _parse_components_recursive(shape_tool, color_tool, components, children_list, step_product_names=None):
    """Recursively parse components of an assembly."""
    if step_product_names is None:
        step_product_names = {}
    
    for i in range(1, components.Length() + 1):
        component_label = components.Value(i)
        component_name = component_label.GetLabelName()
        
        # Get the referred shape (what this component instance points to)
        from OCC.Core.TDF import TDF_Label
        referred_label = TDF_Label()
        is_ref = shape_tool.GetReferredShape(component_label, referred_label)
        
        if is_ref:
            referred_shape = shape_tool.GetShape(referred_label)
            referred_name = referred_label.GetLabelName()
            
            # Try to get a meaningful name using multiple strategies
            display_name = component_name if component_name else referred_name
            
            # Strategy 1: Try TDataStd_Name attribute on both labels
            if not display_name:
                try:
                    from OCC.Core.TDF import Handle_TDF_Attribute
                    attr_handle = Handle_TDF_Attribute()
                    if component_label.FindAttribute(TDataStd_Name.GetID(), attr_handle):
                        if not attr_handle.IsNull():
                            name_attr = TDataStd_Name.DownCast(attr_handle)
                            if name_attr and not name_attr.IsNull():
                                # Fixed: Use ToCString() directly, not Get().ToCString()
                                display_name = name_attr.ToCString()
                    elif referred_label.FindAttribute(TDataStd_Name.GetID(), attr_handle):
                        if not attr_handle.IsNull():
                            name_attr = TDataStd_Name.DownCast(attr_handle)
                            if name_attr and not name_attr.IsNull():
                                # Fixed: Use ToCString() directly, not Get().ToCString()
                                display_name = name_attr.ToCString()
                except (ImportError, AttributeError, TypeError):
                    # Fallback: try GetLabelName
                    if not display_name:
                        try:
                            display_name = component_label.GetLabelName() or referred_label.GetLabelName()
                        except:
                            pass
            
            # Strategy 2: Try alternative name extraction methods
            found_names = []
            if display_name:
                found_names.append(display_name)
            
            # Check both component and referred labels for all name attributes
            for label_to_check in [component_label, referred_label]:
                try:
                    from OCC.Core.TDF import Handle_TDF_Attribute
                    attr_handle = Handle_TDF_Attribute()
                    if label_to_check.FindAttribute(TDataStd_Name.GetID(), attr_handle):
                        if not attr_handle.IsNull():
                            name_attr = TDataStd_Name.DownCast(attr_handle)
                            if name_attr and not name_attr.IsNull():
                                # Fixed: Use ToCString() directly, not Get().ToCString()
                                name_string = name_attr.ToCString()
                                if name_string and name_string not in found_names:
                                    found_names.append(name_string)
                except (ImportError, AttributeError, TypeError):
                    # Fallback: try GetLabelName
                    try:
                        name = label_to_check.GetLabelName()
                        if name and name not in found_names:
                            found_names.append(name)
                    except:
                        pass
            
            # Extract additional metadata including NAUO and PRODUCT information
            component_entry_str = TCollection_AsciiString()
            TDF_Tool.Entry(component_label, component_entry_str)
            component_entry = component_entry_str.ToCString()
            
            referred_entry_str = TCollection_AsciiString()
            TDF_Tool.Entry(referred_label, referred_entry_str)
            referred_entry = referred_entry_str.ToCString()
            
            # Get NAUO (assembly usage occurrence) identifier
            nauo_id = f"NAUO{component_entry.replace(':', '')}"
            
            # Try to extract PRODUCT name from both labels
            product_name = None
            
            # Check referred label first (more likely to have PRODUCT info)
            product_name = _extract_product_name_from_label(shape_tool, referred_label, display_name, step_product_names)
            
            # If no PRODUCT name found, try component label
            if not product_name:
                product_name = _extract_product_name_from_label(shape_tool, component_label, display_name, step_product_names)
            
            # If we found multiple names, see if any look like PRODUCT names or match our extracted list
            if not product_name and len(found_names) > 1:
                # First try to match against our extracted PRODUCT names
                for name in found_names:
                    for entity_id, product_info in step_product_names.items():
                        product_candidate = product_info['name']
                        if (name.lower() == product_candidate.lower() or
                            product_candidate.lower() in name.lower() or
                            name.lower() in product_candidate.lower()):
                            product_name = product_candidate
                            print(f"        🎯 Matched PRODUCT from alternatives: {product_name}")
                            break
                    if product_name:
                        break
                
                # If still no match, look for PRODUCT-like patterns
                if not product_name:
                    for name in found_names:
                        if (len(name) > 10 and 
                            ('_' in name or '-' in name) and
                            not name.startswith('Component_')):
                            product_name = name
                            print(f"        🔍 Selected PRODUCT-like name from alternatives: {product_name}")
                            break
            
            # Use the best available name as display name
            if found_names:
                # Prefer longer, more descriptive names
                display_name = max(found_names, key=lambda x: len(x) if x else 0)
            
            # Last resort: generate a name from entry
            if not display_name:
                entry_str = TCollection_AsciiString()
                TDF_Tool.Entry(component_label, entry_str)
                entry = entry_str.ToCString()
                display_name = f"Component_{entry.replace(':', '_')}"
            
            # Get the location/transformation for this component instance
            component_location = shape_tool.GetLocation(component_label)
            
            # Log the extracted information
            if product_name:
                print(f"        📍 Found component: {display_name} (NAUO: {nauo_id}, PRODUCT: {product_name})")
            else:
                print(f"        📍 Found component: {display_name} (NAUO: {nauo_id})")
            
            component_info = {
                'label': referred_label,  # Use referred label for shape data
                'component_label': component_label,  # Keep component label for transformation
                'shape': referred_shape,
                'name': display_name,
                'nauo_id': nauo_id,  # NAUO identifier
                'product_name': product_name,  # PRODUCT name if found
                'component_entry': component_entry,  # Component label entry
                'referred_entry': referred_entry,  # Referred label entry
                'location': component_location,  # This carries the positioning transformation
                'children': [],
                'is_assembly': shape_tool.IsAssembly(referred_label)
            }
            
            # If the referred shape is also an assembly, recursively get its components
            if shape_tool.IsAssembly(referred_label):
                print(f"            🏭 {display_name} is an assembly")
                sub_components = TDF_LabelSequence()
                shape_tool.GetComponents(referred_label, sub_components)
                if sub_components.Length() > 0:
                    print(f"            🔄 Processing {sub_components.Length()} sub-components")
                    _parse_components_recursive(shape_tool, color_tool, sub_components, component_info['children'], step_product_names)
            else:
                print(f"            🔧 {display_name} is a simple part")
            
            children_list.append(component_info)
