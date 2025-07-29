"""Utility functions for reporting and analysis."""


def print_component_metadata_report(hierarchy, depth=0):
    """Print a detailed report of all components with their metadata."""
    indent = "  " * depth
    
    for shape_info in hierarchy:
        name = shape_info.get('name', 'Unnamed')
        nauo_id = shape_info.get('nauo_id', 'N/A')
        product_name = shape_info.get('product_name', 'N/A')
        component_entry = shape_info.get('component_entry', 'N/A')
        referred_entry = shape_info.get('referred_entry', 'N/A')
        is_assembly = shape_info.get('is_assembly', False)
        
        print(f"{indent}📦 {'Assembly' if is_assembly else 'Part'}: {name}")
        print(f"{indent}   🏷️  NAUO ID: {nauo_id}")
        print(f"{indent}   🏭 PRODUCT: {product_name}")
        print(f"{indent}   📍 Component Entry: {component_entry}")
        print(f"{indent}   🔗 Referred Entry: {referred_entry}")
        
        if shape_info.get('children'):
            print(f"{indent}   📁 Children ({len(shape_info['children'])}):")
            print_component_metadata_report(shape_info['children'], depth + 2)
        print()


def count_shapes_in_hierarchy(shape_info):
    """Count total number of shapes in a hierarchy."""
    count = 1  # Count this shape
    for child in shape_info['children']:
        count += count_shapes_in_hierarchy(child)
    return count
