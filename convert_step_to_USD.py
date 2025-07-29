#!/usr/bin/env python3
"""
STEP to USD Converter with Hierarchy Preservation

Main entry point for converting STEP files to USD format while preserving
the hierarchical structure and metadata.
"""

from pxr import Usd, UsdGeom
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.TopLoc import TopLoc_Location

from step_convert.name_utils import extract_product_names_from_step_file
from step_convert.step_reader import read_step_file_with_hierarchy
from step_convert.usd_converter import convert_hierarchical_shape_to_usd, convert_hierarchical_shape_to_usd_parallel
from step_convert.utils import print_component_metadata_report, count_shapes_in_hierarchy


def main(step_path=None, usd_path=None, use_parallel=True, max_workers=None):
        
    print("=" * 60)
    print("🚀 STEP to USD Converter with Hierarchy")
    print("=" * 60)
    
    # Read STEP file with hierarchical structure
    print(f"🎯 Target STEP file: {step_path}")
    print(f"🎯 Output USD file: {usd_path}")
    if use_parallel:
        print(f"⚡ Parallel processing: Enabled (max_workers={max_workers})")
    else:
        print("🔄 Parallel processing: Disabled (sequential)")
    print()
    
    # First, extract PRODUCT names directly from STEP file
    print("🔍 Extracting PRODUCT names from STEP file...")
    step_product_names = extract_product_names_from_step_file(step_path)
    print()
    
    try:
        hierarchy, shape_tool, _ = read_step_file_with_hierarchy(step_path, step_product_names)
    except Exception as e:
        print(f"❌ Error reading STEP file with hierarchy: {e}")
        print("🔄 Falling back to simple shape reading...")
        # Fallback to simple reading
        shape = read_step_file(step_path)
        if not shape:
            raise ValueError("❌ No shapes found in STEP file")
        
        # Create a simple hierarchy structure
        hierarchy = [{
            'shape': shape,
            'name': 'Root_Assembly',
            'location': TopLoc_Location(),
            'children': []
        }]
        shape_tool = None
        print("✅ Fallback successful")
    
    if not hierarchy:
        raise ValueError("❌ No shapes found in STEP file")
    
    total_hierarchy_shapes = sum(count_shapes_in_hierarchy(shape_info) for shape_info in hierarchy)
    print(f"📊 Found {len(hierarchy)} root-level assemblies/parts")
    print(f"📊 Total shapes in hierarchy: {total_hierarchy_shapes}")
    print()
    
    # Print detailed metadata report
    print("=" * 60)
    print("📋 COMPONENT METADATA REPORT")
    print("=" * 60)
    print_component_metadata_report(hierarchy)
    print("=" * 60)
    print()

    # Create USD stage
    print("🏗️ Creating USD stage...")
    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # Root Xform
    root = UsdGeom.Xform.Define(stage, "/Root")
    print("✅ USD stage initialized")
    print()

    # Convert hierarchical shapes
    print("🔄 Converting shapes to USD...")
    print("-" * 40)
    total_shapes = 0
    
    if use_parallel:
        # Use parallel mesh processing
        convert_hierarchical_shape_to_usd_parallel(stage, root, hierarchy, shape_tool, max_workers)
        
        # Count total shapes for reporting
        for shape_info in hierarchy:
            total_shapes += count_shapes_in_hierarchy(shape_info)
        
        print(f"✅ Parallel conversion complete")
    else:
        # Use sequential processing (original method)
        for i, shape_info in enumerate(hierarchy):
            print(f"🏗️ Converting assembly {i+1}/{len(hierarchy)}: {shape_info['name']}")
            
            # Convert the hierarchical shape
            convert_hierarchical_shape_to_usd(stage, root, shape_info, shape_tool)
            
            # Count shapes recursively
            shapes_in_assembly = count_shapes_in_hierarchy(shape_info)
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
    print("=" * 60)


if __name__ == "__main__":
    step_file = "./Mk10.STEP"
    # step_file = "./test_files/nist_stc_10_asme1_ap242-e2.stp"
    usd_file = "output.usda"
    
    # Configuration for parallel processing
    use_parallel = True  # Set to False to use sequential processing
    max_workers = None   # None = auto-detect CPU cores, or set specific number like 4

    try:
        main(step_file, usd_file, use_parallel, max_workers)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Exiting gracefully.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Check the STEP file path and try again.")
