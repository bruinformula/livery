"""STEP file reading and hierarchy parsing functionality."""

from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.XCAFApp import XCAFApp_Application
from OCC.Core.TDataStd import TDataStd_Name
from OCC.Core.TDF import TDF_LabelSequence, TDF_Tool, TDF_AttributeIterator
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TCollection import TCollection_AsciiString
from OCC.Core.TDF import TDF_Label

def read_step_file_with_hierarchy(step_file_path, step_product_names=None):
    """Read STEP file and return hierarchical structure with transformations."""
    if step_product_names is None:
        step_product_names = {}
        
    print("Initializing XCAF document...")
    app = XCAFApp_Application.GetApplication()
    doc = TDocStd_Document("XCAF")
    app.NewDocument("STEP", doc)
    
    print("Setting up XCAF tools...")
    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool(doc.Main())
    
    print(f"Reading STEP file: {step_file_path}")
    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetLayerMode(True)
    reader.SetNameMode(True)
    reader.SetMatMode(True)
    
    print("Processing STEP file...")
    status = reader.ReadFile(step_file_path)
    if status != 1:
        raise ValueError(f"Error reading STEP file: {step_file_path}")
    
    print("Transferring data to XCAF document...")
    reader.Transfer(doc)
    
    print("Analyzing hierarchical structure...")
    labels = TDF_LabelSequence()
    shape_tool.GetShapes(labels)
    
    if labels.Length() == 0:
        raise ValueError("No shapes found in STEP file")
    
    print(f"Found {labels.Length()} labels at root")
    
    root_label = labels.Value(1)
    root_name = root_label.GetLabelName()
    if not root_name:
        root_name = "Top"
    
    print(f"Root assembly: {root_name}")
    
    hierarchy = []
    if shape_tool.IsAssembly(root_label):
        print(f"    Root is an assembly")
        root_shape = shape_tool.GetShape(root_label)
        
        root_info = {
            'label': root_label,
            'shape': root_shape,
            'name': root_name,
            'location': TopLoc_Location(),
            'children': [],
            'is_assembly': True
        }
        
        components = TDF_LabelSequence()
        shape_tool.GetComponents(root_label, components)
        print(f"    Root assembly has {components.Length()} components")
        
        parse_components_recursive(shape_tool, color_tool, components, root_info['children'], step_product_names)
        
        hierarchy.append(root_info)
    else:
        print(f"    Root is a simple shape")
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
    
    print(f"Hierarchy analysis complete!")
    return hierarchy, shape_tool, color_tool


def extract_product_name_from_label(label, default_name, step_product_names=None):
    """Extract PRODUCT name from STEP label attributes using multiple strategies."""
    if step_product_names is None:
        step_product_names = {}
        
    try:
        label_entry_str = TCollection_AsciiString()
        TDF_Tool.Entry(label, label_entry_str)
        label_entry = label_entry_str.ToCString()
        
        for entity_id, product_info in step_product_names.items():
            product_name = product_info['name']
            if (default_name and 
                (default_name.lower() == product_name.lower() or
                 product_name.lower() in default_name.lower() or
                 default_name.lower() in product_name.lower())):
                print(f"        🎯 Matched PRODUCT by name: {product_name}")
                return product_name
            
            if entity_id in label_entry:
                print(f"        Matched PRODUCT by entity ID #{entity_id}: {product_name}")
                return product_name
        
        found_names = []
        
        try:
            name_attr = TDataStd_Name()
            if label.FindAttribute(TDataStd_Name.GetID(), name_attr):
                if not name_attr.IsNull():
                    name_string = name_attr.Get().ToCString()
                    if name_string and name_string != default_name:
                        found_names.append(name_string)
        except (ImportError, AttributeError, TypeError) as e:
            print(f"        Warning: Error extracting PRODUCT name: {e}")
            try:
                name = label.GetLabelName()
                if name and name != default_name:
                    found_names.append(name)
            except:
                pass
        
        attr_iter = TDF_AttributeIterator()
        attr_iter.Initialize(label, True)
        
        while attr_iter.More():
            attr = attr_iter.Value()
            
            if attr.DynamicType().Name() == 'TDataStd_Name':
                name_attr = TDataStd_Name.DownCast(attr)
                if name_attr:
                    name_string = name_attr.Get().ToCString()
                    if name_string and name_string != default_name and name_string not in found_names:
                        found_names.append(name_string)
            
            attr_iter.Next()
        
        parent_label = label.Father()
        if not parent_label.IsNull():
            try:
                name_attr = TDataStd_Name()
                if parent_label.FindAttribute(TDataStd_Name.GetID(), name_attr):
                    if not name_attr.IsNull():
                        parent_name_string = name_attr.Get().ToCString()
                        if parent_name_string and parent_name_string != default_name and parent_name_string not in found_names:
                            found_names.append(parent_name_string)
            except (ImportError, AttributeError, TypeError):
                try:
                    parent_name = parent_label.GetLabelName()
                    if parent_name and parent_name != default_name and parent_name not in found_names:
                        found_names.append(parent_name)
                except:
                    pass
        
        for i in range(1, label.NbChildren() + 1):
            child_label = label.FindChild(i)
            if not child_label.IsNull():
                try:
                    name_attr = TDataStd_Name()
                    if child_label.FindAttribute(TDataStd_Name.GetID(), name_attr):
                        if not name_attr.IsNull():
                            child_name_string = name_attr.Get().ToCString()
                            if child_name_string and child_name_string != default_name and child_name_string not in found_names:
                                found_names.append(child_name_string)
                except (ImportError, AttributeError, TypeError):
                    try:
                        child_name = child_label.GetLabelName()
                        if child_name and child_name != default_name and child_name not in found_names:
                            found_names.append(child_name)
                    except:
                        pass
        
        if found_names:
            for name in found_names:
                if (len(name) > 10 and
                    ('_' in name or '-' in name) and
                    not name.startswith('Component_') and
                    not name.startswith('Shape_')):
                    print(f"        🔍 Found PRODUCT-like name: {name}")
                    return name
            
            longest_name = max(found_names, key=len)
            if len(longest_name) > len(default_name or ""):
                print(f"        🔍 Using longest found name: {longest_name}")
                return longest_name
        
        if (default_name and 
            len(default_name) > 5 and
            ('_' in default_name or '-' in default_name) and not default_name.startswith('Component_')):
            print(f"        Default name looks like PRODUCT: {default_name}")
            return default_name
            
    except Exception as e:
        print(f"        Warning: Error extracting PRODUCT name: {e}")
    
    return None


def extract_component_names(shape_tool, component_label, referred_label, step_product_names=None):
    """Extract display name and product name from component and referred labels."""
    if step_product_names is None:
        step_product_names = {}
    
    component_name = component_label.GetLabelName()
    referred_name = referred_label.GetLabelName()
    
    display_name = component_name if component_name else referred_name
    
    if not display_name:
        try:
            name_attr = TDataStd_Name()
            if component_label.FindAttribute(TDataStd_Name.GetID(), name_attr):
                if not name_attr.IsNull():
                    display_name = name_attr.Get().ToCString()
            elif referred_label.FindAttribute(TDataStd_Name.GetID(), name_attr):
                if not name_attr.IsNull():
                    display_name = name_attr.Get().ToCString()
        except (ImportError, AttributeError, TypeError):
            if not display_name:
                try:
                    display_name = component_label.GetLabelName() or referred_label.GetLabelName()
                except:
                    pass
    
    found_names = []
    if display_name:
        found_names.append(display_name)
    
    for label_to_check in [component_label, referred_label]:
        try:
            name_attr = TDataStd_Name()
            if label_to_check.FindAttribute(TDataStd_Name.GetID(), name_attr):
                if not name_attr.IsNull():
                    name_string = name_attr.Get().ToCString()
                    if name_string and name_string not in found_names:
                        found_names.append(name_string)
        except (ImportError, AttributeError, TypeError):
            try:
                name = label_to_check.GetLabelName()
                if name and name not in found_names:
                    found_names.append(name)
            except:
                pass
    
    product_name = None
    
    product_name = extract_product_name_from_label(referred_label, display_name, step_product_names)
    
    if not product_name:
        product_name = extract_product_name_from_label(component_label, display_name, step_product_names)
    
    if not product_name and len(found_names) > 1:
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
        
        if not product_name:
            for name in found_names:
                if (len(name) > 10 and 
                    ('_' in name or '-' in name) and
                    not name.startswith('Component_')):
                    product_name = name
                    print(f"        🔍 Selected PRODUCT-like name from alternatives: {product_name}")
                    break
    
    if found_names:
        display_name = max(found_names, key=lambda x: len(x) if x else 0)
    
    if not display_name:
        entry_str = TCollection_AsciiString()
        TDF_Tool.Entry(component_label, entry_str)
        entry = entry_str.ToCString()
        display_name = f"Component_{entry.replace(':', '_')}"
    
    return display_name, product_name


def parse_components_recursive(shape_tool, color_tool, components, children_list, step_product_names=None):
    """Recursively parse components of an assembly."""
    if step_product_names is None:
        step_product_names = {}
    
    for i in range(1, components.Length() + 1):
        component_label = components.Value(i)
        
        referred_label = TDF_Label()
        is_ref = shape_tool.GetReferredShape(component_label, referred_label)
        
        if is_ref:
            referred_shape = shape_tool.GetShape(referred_label)
            
            display_name, product_name = extract_component_names(shape_tool, component_label, referred_label, step_product_names)
            
            component_entry_str = TCollection_AsciiString()
            TDF_Tool.Entry(component_label, component_entry_str)
            component_entry = component_entry_str.ToCString()
            
            referred_entry_str = TCollection_AsciiString()
            TDF_Tool.Entry(referred_label, referred_entry_str)
            referred_entry = referred_entry_str.ToCString()
            
            nauo_id = f"NAUO{component_entry.replace(':', '')}"
            
            component_location = shape_tool.GetLocation(component_label)
            
            if product_name:
                print(f"        Found component: {display_name} (NAUO: {nauo_id}, PRODUCT: {product_name})")
            else:
                print(f"        Found component: {display_name} (NAUO: {nauo_id})")
            
            component_info = {
                'label': referred_label,
                'component_label': component_label,
                'shape': referred_shape,
                'name': display_name,
                'nauo_id': nauo_id,
                'product_name': product_name,
                'component_entry': component_entry,
                'referred_entry': referred_entry,
                'location': component_location,
                'children': [],
                'is_assembly': shape_tool.IsAssembly(referred_label)
            }
            
            if shape_tool.IsAssembly(referred_label):
                print(f"            {display_name} is an assembly")
                sub_components = TDF_LabelSequence()
                shape_tool.GetComponents(referred_label, sub_components)
                if sub_components.Length() > 0:
                    print(f"            Processing {sub_components.Length()} sub-components")
                    parse_components_recursive(shape_tool, color_tool, sub_components, component_info['children'], step_product_names)
            else:
                print(f"            {display_name} is a simple part")
            
            children_list.append(component_info)
