"""STEP file reading and hierarchy parsing functionality."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.XCAFApp import XCAFApp_Application
from OCC.Core.TDF import TDF_LabelSequence, TDF_Tool, TDF_AttributeIterator
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TCollection import TCollection_AsciiString
from OCC.Core.TDF import TDF_Label
from OCC.Core.TopoDS import TopoDS_Shape


@dataclass
class ProductInfo:
    """Information about a STEP product."""
    name: str
    description: Optional[str] = None
    part_number: Optional[str] = None
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format for backward compatibility."""
        result = {'name': self.name}
        if self.description:
            result['description'] = self.description
        if self.part_number:
            result['part_number'] = self.part_number
        return result

@dataclass
class ComponentInfo:
    """Information about a component in the STEP hierarchy."""
    label: TDF_Label
    shape: TopoDS_Shape
    name: str
    location: TopLoc_Location
    children: List[ComponentInfo]
    is_assembly: bool
    component_label: Optional[TDF_Label] = None
    nauo_id: Optional[str] = None
    product_name: Optional[str] = None
    component_entry: Optional[str] = None
    referred_entry: Optional[str] = None

# Type aliases
ProductDict = Dict[str, ProductInfo]

class STEPFile:
    step_file_path: Path
    hierarchy: List[ComponentInfo]
    shape_tool: Any  # XCAFDoc_ShapeTool 
    color_tool: Any  # XCAFDoc_ColorTool

    def __init__(self, step_file_path: Union[str, Path], step_product_names: Optional[ProductDict] = None):
        hierarchy, shape_tool, color_tool = STEPFile.read_step_file_with_hierarchy(step_file_path, step_product_names)

        if not hierarchy:
            raise ValueError("No shapes found in STEP file")
        
        self.hierarchy = hierarchy
        self.shape_tool = shape_tool
        self.color_tool = color_tool


    def print_component_metadata_report(self, depth=0, hierarchy = None):
        """Print a detailed report of all components with their metadata."""
        indent = "  " * depth
        
        if hierarchy == None:
            hierarchy = self.hierarchy

        for shape_info in hierarchy:
            name = shape_info.name
            nauo_id = shape_info.nauo_id or 'N/A'
            product_name = shape_info.product_name or 'N/A'
            component_entry = shape_info.component_entry or 'N/A'
            referred_entry = shape_info.referred_entry or 'N/A'
            is_assembly = shape_info.is_assembly
            
            print(f"{indent}📦 {'Assembly' if is_assembly else 'Part'}: {name}")
            print(f"{indent}   🏷️  NAUO ID: {nauo_id}")
            print(f"{indent}   🏭 PRODUCT: {product_name}")
            print(f"{indent}   📍 Component Entry: {component_entry}")
            print(f"{indent}   🔗 Referred Entry: {referred_entry}")
            
            if shape_info.children:
                print(f"{indent}   📁 Children ({len(shape_info.children)}):")
                self.print_component_metadata_report(depth + 2, shape_info.children)
            print()

    def print_total_hierarchy_shapes(self):
        total_hierarchy_shapes = sum(self.count_shapes_in_hierarchy(shape_info) for shape_info in self.hierarchy)
        print(f"Found {len(self.hierarchy)} root-level assemblies/parts")
        print(f"Total shapes in hierarchy: {total_hierarchy_shapes}")

    @staticmethod
    def count_shapes_in_hierarchy(shape_info):
        """Count total number of shapes in a hierarchy."""
        count = 1  # Count this shape

        for child in shape_info.children:
            count += STEPFile.count_shapes_in_hierarchy(child)
        return count


    @staticmethod
    def read_step_file_with_hierarchy(step_file_path: Union[str, Path], step_product_names: Optional[ProductDict] = None) -> Tuple[List[ComponentInfo], Any, Any]:
        """Read STEP file and return hierarchical structure with transformations."""
        if step_product_names is None:
            step_product_names = {}
            
        print("Initializing XCAF document...")
        app: Any = XCAFApp_Application.GetApplication()
        doc: TDocStd_Document = TDocStd_Document("XCAF")
        app.NewDocument("STEP", doc)
        
        print("Setting up XCAF tools...")
        shape_tool: Any = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
        color_tool: Any = XCAFDoc_DocumentTool.ColorTool(doc.Main())
        
        print(f"Reading STEP file: {step_file_path}")
        reader: STEPCAFControl_Reader = STEPCAFControl_Reader()
        reader.SetColorMode(True)
        reader.SetLayerMode(True)
        reader.SetNameMode(True)
        reader.SetMatMode(True)
        
        print("Processing STEP file...")
        status: int = reader.ReadFile(step_file_path)
        if status != 1:
            raise ValueError(f"Error reading STEP file: {step_file_path}")
        
        print("Transferring data to XCAF document...")
        reader.Transfer(doc)
        
        print("Analyzing hierarchical structure...")
        labels: TDF_LabelSequence = TDF_LabelSequence()
        shape_tool.GetShapes(labels)
        
        if labels.Length() == 0:
            raise ValueError("No shapes found in STEP file")
        
        print(f"Found {labels.Length()} labels at root")
        
        root_label: TDF_Label = labels.Value(1)
        root_name: Optional[str] = root_label.GetLabelName()
        if not root_name:
            root_name = "Top"
        
        print(f"Root assembly: {root_name}")
        
        hierarchy: List[ComponentInfo] = []

        if shape_tool.IsAssembly(root_label):
            print(f"    Root is an assembly")
            root_shape: TopoDS_Shape = shape_tool.GetShape(root_label)
            
            root_info = ComponentInfo(
                label=root_label,
                shape=root_shape,
                name=root_name,
                location=TopLoc_Location(),
                children=[],
                is_assembly=True
            )
            
            components: TDF_LabelSequence = TDF_LabelSequence()
            shape_tool.GetComponents(root_label, components)
            print(f"    Root assembly has {components.Length()} components")
            
            STEPFile.parse_components_recursive(shape_tool, color_tool, components, root_info.children, step_product_names)
            
            hierarchy.append(root_info)
        else:
            print(f"    Root is a simple shape")
            root_shape: TopoDS_Shape = shape_tool.GetShape(root_label)
            root_info = ComponentInfo(
                label=root_label,
                shape=root_shape,
                name=root_name,
                location=TopLoc_Location(),
                children=[],
                is_assembly=False
            )
            hierarchy.append(root_info)
        
        print(f"Hierarchy analysis complete!")
        return hierarchy, shape_tool, color_tool

    @staticmethod
    def extract_product_name_from_label(label: TDF_Label, default_name: Optional[str], step_product_names: Optional[ProductDict] = None) -> Optional[str]:
        """Extract PRODUCT name from STEP label attributes using multiple strategies."""
        if step_product_names is None:
            step_product_names = {}
            
        try:
            label_entry_str: TCollection_AsciiString = TCollection_AsciiString()
            TDF_Tool.Entry(label, label_entry_str)
            label_entry: str = label_entry_str.ToCString()
            
            for entity_id, product_info in step_product_names.items():
                product_name: str = product_info.name
                if (default_name and 
                    (default_name.lower() == product_name.lower() or
                    product_name.lower() in default_name.lower() or
                    default_name.lower() in product_name.lower())):
                    print(f"        🎯 Matched PRODUCT by name: {product_name}")
                    return product_name
                
                if entity_id in label_entry:
                    print(f"        Matched PRODUCT by entity ID #{entity_id}: {product_name}")
                    return product_name
            
            found_names: List[str] = []
            
            try:
                name: Optional[str] = label.GetLabelName()
                if name and name != default_name:
                    found_names.append(name)
            except Exception as e:
                print(f"        Warning: Error extracting PRODUCT name from GetLabelName: {e}")
                pass
            
            attr_iter: TDF_AttributeIterator = TDF_AttributeIterator()
            attr_iter.Initialize(label, True)
            
            while attr_iter.More():
                attr: Any = attr_iter.Value()
                
                # Skip TDataStd_Name attributes as they cause issues in this OpenCASCADE version
                # Use GetLabelName() instead
                
                attr_iter.Next()
            
            parent_label: TDF_Label = label.Father()
            if not parent_label.IsNull():
                try:
                    parent_name: Optional[str] = parent_label.GetLabelName()
                    if parent_name and parent_name != default_name and parent_name not in found_names:
                        found_names.append(parent_name)
                except Exception:
                    pass
            
            for i in range(1, label.NbChildren() + 1):
                child_label: TDF_Label = label.FindChild(i)
                if not child_label.IsNull():
                    try:
                        child_name: Optional[str] = child_label.GetLabelName()
                        if child_name and child_name != default_name and child_name not in found_names:
                            found_names.append(child_name)
                    except Exception:
                        pass
            
            if found_names:
                for name in found_names:
                    if (len(name) > 10 and
                        ('_' in name or '-' in name) and
                        not name.startswith('Component_') and
                        not name.startswith('Shape_')):
                        print(f"        🔍 Found PRODUCT-like name: {name}")
                        return name
                
                longest_name: str = max(found_names, key=len)
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

    @staticmethod
    def extract_component_names(component_label: TDF_Label, referred_label: TDF_Label, step_product_names: Optional[ProductDict] = None) -> Tuple[str, Optional[str]]:
        """Extract display name and product name from component and referred labels."""
        if step_product_names is None:
            step_product_names = {}
        
        component_name: Optional[str] = component_label.GetLabelName()
        referred_name: Optional[str] = referred_label.GetLabelName()
        
        display_name: Optional[str] = component_name if component_name else referred_name
        
        if not display_name:
            try:
                display_name = component_label.GetLabelName() or referred_label.GetLabelName()
            except Exception:
                pass
        
        found_names: List[str] = []
        if display_name:
            found_names.append(display_name)
        
        for label_to_check in [component_label, referred_label]:
            try:
                name: Optional[str] = label_to_check.GetLabelName()
                if name and name not in found_names:
                    found_names.append(name)
            except Exception:
                pass
        
        product_name: Optional[str] = None
        
        product_name = STEPFile.extract_product_name_from_label(referred_label, display_name, step_product_names)
        
        if not product_name:
            product_name = STEPFile.extract_product_name_from_label(component_label, display_name, step_product_names)
        
        if not product_name and len(found_names) > 1:
            for name in found_names:
                for entity_id, product_info in step_product_names.items():
                    product_candidate: str = product_info.name
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
            entry_str: TCollection_AsciiString = TCollection_AsciiString()
            TDF_Tool.Entry(component_label, entry_str)
            entry: str = entry_str.ToCString()
            display_name = f"Component_{entry.replace(':', '_')}"
        
        return display_name, product_name

    @staticmethod
    def parse_components_recursive(shape_tool: Any, color_tool: Any, components: TDF_LabelSequence, children_list: List[ComponentInfo], step_product_names: Optional[ProductDict] = None) -> None:
        """Recursively parse components of an assembly."""
        if step_product_names is None:
            step_product_names = {}
        
        for i in range(1, components.Length() + 1):
            component_label: TDF_Label = components.Value(i)
            
            referred_label: TDF_Label = TDF_Label()
            is_ref: bool = shape_tool.GetReferredShape(component_label, referred_label)
            
            if is_ref:
                referred_shape: TopoDS_Shape = shape_tool.GetShape(referred_label)
                
                display_name, product_name = STEPFile.extract_component_names(component_label, referred_label, step_product_names)
                
                component_entry_str: TCollection_AsciiString = TCollection_AsciiString()
                TDF_Tool.Entry(component_label, component_entry_str)
                component_entry: str = component_entry_str.ToCString()
                
                referred_entry_str: TCollection_AsciiString = TCollection_AsciiString()
                TDF_Tool.Entry(referred_label, referred_entry_str)
                referred_entry: str = referred_entry_str.ToCString()
                
                nauo_id: str = f"NAUO{component_entry.replace(':', '')}"
                
                component_location: TopLoc_Location = shape_tool.GetLocation(component_label)
                
                if product_name:
                    print(f"        Found component: {display_name} (NAUO: {nauo_id}, PRODUCT: {product_name})")
                else:
                    print(f"        Found component: {display_name} (NAUO: {nauo_id})")
                
                component_info = ComponentInfo(
                    label=referred_label,
                    shape=referred_shape,
                    name=display_name,
                    location=component_location,
                    children=[],
                    is_assembly=shape_tool.IsAssembly(referred_label),
                    component_label=component_label,
                    nauo_id=nauo_id,
                    product_name=product_name,
                    component_entry=component_entry,
                    referred_entry=referred_entry
                )
                
                if shape_tool.IsAssembly(referred_label):
                    print(f"            {display_name} is an assembly")
                    sub_components: TDF_LabelSequence = TDF_LabelSequence()
                    shape_tool.GetComponents(referred_label, sub_components)
                    if sub_components.Length() > 0:
                        print(f"            Processing {sub_components.Length()} sub-components")
                        STEPFile.parse_components_recursive(shape_tool, color_tool, sub_components, component_info.children, step_product_names)
                else:
                    print(f"            {display_name} is a simple part")
                
                children_list.append(component_info)
