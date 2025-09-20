"""Name utilities for STEP file processing."""

import re
import uuid
from typing import Dict
from .step_reader import ProductInfo

_used_names = set()

def sanitize_usd_name(name):
    """Convert a STEP name to a valid USD prim name."""
    if not name:
        return "unnamed"
    
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[^\w]', '_', name)
    
    # Ensure it doesn't start with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = f"prim_{sanitized}"
    
    if not sanitized:
        sanitized = "unnamed"
    
    return sanitized

def generate_unique_name():
    """Generate a unique name for unnamed components."""
    while True:
        name = f"component_{uuid.uuid4().hex[:8]}"
        if name not in _used_names:
            _used_names.add(name)
            return name

def extract_product_names_from_step_file(step_file_path) -> Dict[str, ProductInfo]:
    """Extract PRODUCT names directly from STEP file text."""
    product_names: Dict[str, ProductInfo] = {}
    try:
        with open(step_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # PRODUCT ( 'name', 'description', '', ( #id ) ) ;
        product_pattern = r"PRODUCT\s*\(\s*'([^']+)'\s*,\s*'([^']*)'\s*,\s*'[^']*'\s*,\s*\(\s*#(\d+)\s*\)\s*\)\s*;"
        
        matches = re.findall(product_pattern, content, re.IGNORECASE)
        
        for match in matches:
            name, description, entity_id = match
            product_names[entity_id] = ProductInfo(
                name=name,
                description=description if description else None
            )
        
        print(f"Extracted {len(product_names)} PRODUCT entries from STEP file")
        return product_names
        
    except Exception as e:
        print(f"Warning: Could not extract PRODUCT names from STEP file: {e}")
        return {}
