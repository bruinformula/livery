#!/usr/bin/env python3
"""
STEP to USD Converter with Hierarchy Preservation

Main entry point for converting STEP files to USD format while preserving
the hierarchical structure and metadata.
"""

from pxr import Usd, UsdGeom

from .name_utils import extract_product_names_from_step_file
from .step_reader import STEPFile
from .usd_converter import convert_hierarchical_shape_to_usd, clear_geometry_registry
from .usd_converter import _reference_counter


def convert_step_to_usd(step_path=None, usd_path=None):
        
    print("=" * 60)
    print("🚀 STEP to USD Converter with Hierarchy")
    print("=" * 60)
    
    # Read STEP file with hierarchical structure
    print(f"🎯 Target STEP file: {step_path}")
    print(f"🎯 Output USD file: {usd_path}")
    
    # First, extract PRODUCT names directly from STEP file
    print("🔍 Extracting PRODUCT names from STEP file...")
    step_product_names = extract_product_names_from_step_file(step_path)
    print()
    
    try:
        step_file = STEPFile(step_path, step_product_names)
    except Exception as e:
        print(f"Error reading STEP file with hierarchy: {e}")
        print("Exiting...x")
        return False
    
    # Print detailed metadata report
    print("=" * 60)
    print("COMPONENT METADATA REPORT")
    step_file.print_total_hierarchy_shapes()
    print("=" * 60)
    step_file.print_component_metadata_report()
    print("=" * 60)
    print()

    # Create USD stage
    print("🏗️ Creating USD stage...")
    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # Clear geometry registry for new conversion
    clear_geometry_registry()

    # Root Xform
    root = UsdGeom.Xform.Define(stage, "/")
    print("✅ USD stage initialized")
    print()

    # Convert hierarchical shapes
    print("🔄 Converting shapes to USD with reference detection...")
    print("-" * 40)
    total_shapes = 0

    for i, shape_info in enumerate(step_file.hierarchy):
        print(f"🏗️ Converting assembly {i+1}/{len(step_file.hierarchy)}: {shape_info.name}")
        
        # Convert the hierarchical shape
        convert_hierarchical_shape_to_usd(stage, root, shape_info, step_file.shape_tool)
        
        # Count shapes recursively
        shapes_in_assembly = STEPFile.count_shapes_in_hierarchy(shape_info)
        total_shapes += shapes_in_assembly
        print(f"✅ Assembly complete ({shapes_in_assembly} shapes)")
    print()

    print("-" * 40)
    print("💾 Saving USD file...")
    stage.GetRootLayer().Save()
    print()
    print("=" * 60)
    print("✅ CONVERSION COMPLETE!")
    print(f"📁 USD file saved: {usd_path}")
    print(f"📊 Total shapes exported: {total_shapes}")
    print(f"🏗️ Hierarchy preserved: ✅")
    
    # Print reference statistics
    if _reference_counter:
        print(f"🔗 USD References created:")
        for referred_entry, count in _reference_counter.items():
            if count > 1:
                print(f"    {referred_entry}: {count} instances")
    
    print("=" * 60)