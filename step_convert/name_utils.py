"""Utilities for name processing and validation."""

import re
import uuid


def sanitize_usd_name(name):
    """Sanitize a name to be valid for USD prim names."""
    if not name:
        return "unnamed"
    
    # Replace invalid characters with underscores
    # USD names can only contain letters, numbers, and underscores
    # and must start with a letter or underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    
    # Ensure it starts with a letter or underscore
    if sanitized and not (sanitized[0].isalpha() or sanitized[0] == '_'):
        sanitized = 'P_' + sanitized
    
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove trailing underscores
    sanitized = sanitized.rstrip('_')
    
    # Ensure it's not empty after sanitization
    if not sanitized:
        return "unnamed"
    
    return sanitized


def generate_unique_name(base_name="UnnamedShape"):
    """Generate a unique name with UUID suffix."""
    return f"{base_name}_{uuid.uuid4().hex[:8]}"


def extract_product_names_from_step_file(step_file_path):
    """Extract PRODUCT names directly from STEP file text."""
    product_names = {}
    try:
        with open(step_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Find PRODUCT entries using regex
        # PRODUCT ( 'name', 'description', '', ( #id ) ) ;
        product_pattern = r"PRODUCT\s*\(\s*'([^']+)'\s*,\s*'([^']*)'\s*,\s*'[^']*'\s*,\s*\(\s*#(\d+)\s*\)\s*\)\s*;"
        
        matches = re.findall(product_pattern, content, re.IGNORECASE)
        
        for match in matches:
            name, description, entity_id = match
            product_names[entity_id] = {
                'name': name,
                'description': description,
                'entity_id': entity_id
            }
            print(f"    🏭 Found STEP PRODUCT: '{name}' (ID: #{entity_id})")
        
        print(f"📊 Extracted {len(product_names)} PRODUCT entries from STEP file")
        return product_names
        
    except Exception as e:
        print(f"⚠️ Warning: Could not extract PRODUCT names from STEP file: {e}")
        return {}
