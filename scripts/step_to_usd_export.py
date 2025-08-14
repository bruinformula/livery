#!/usr/bin/env python3
"""
STEP to USD Converter - Command Line Interface

A command-line tool for converting STEP files to USD format while preserving
hierarchical structure and metadata.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the parent directory to Python path to import step_convert
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from bruin_livery.step_convert.step_to_usd import convert_step_to_usd
from bruin_livery.step_convert import config

def setup_config(args):
    """Update configuration based on command line arguments."""
    if args.linear_deflection is not None:
        config.MESH_LINEAR_DEFLECTION = args.linear_deflection
    
    if args.angular_deflection is not None:
        config.MESH_ANGULAR_DEFLECTION = args.angular_deflection
    
    if args.flip_normals:
        print("normal")
        config.FLIP_NORMALS = args.flip_normals
    
    if args.force_winding:
        print("force")
        config.FORCE_WINDING_ORDER = args.force_winding

def main():
    """Main command line interface."""
    parser = argparse.ArgumentParser(
        description="Convert STEP files to USD format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.step output.usd
  %(prog)s --linear 0.5 --angular 0.2 input.step output.usd
  %(prog)s --flip-normals input.step output.usd
  %(prog)s input.step output.usd
        """
    )
    
    #required
    parser.add_argument(
        "input_step",
        help="Input STEP file path"
    )
    
    parser.add_argument(
        "output_usd", 
        help="Output USD file path"
    )
    
    #optionals
    parser.add_argument(
        "-l", "--linear",
        type=float,
        dest="linear_deflection",
        help=f"Linear deflection for tessellation (default: {config.MESH_LINEAR_DEFLECTION})"
    )
    
    parser.add_argument(
        "-a", "--angular", 
        type=float,
        dest="angular_deflection",
        help=f"Angular deflection for tessellation (default: {config.MESH_ANGULAR_DEFLECTION})"
    )
    
    parser.add_argument(
        "--flip-normals",
        action="store_true",
        help="Flip normal directions"
    )
    
    parser.add_argument(
        "--force-winding",
        action="store_true",
        help="Force consistent winding order for faces"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_step)
    if not input_path.exists():
        print(f"❌ Error: Input STEP file does not exist: {args.input_step}")
        sys.exit(1)
    
    if not input_path.suffix.lower() in ['.step', '.stp']:
        print(f"⚠️  Warning: Input file doesn't have .step or .stp extension: {args.input_step}")
    
    # Validate output directory
    output_path = Path(args.output_usd)
    if not output_path.parent.exists():
        print(f"❌ Error: Output directory does not exist: {output_path.parent}")
        sys.exit(1)
    
    # Update configuration
    setup_config(args)
    
    # Perform conversion
    success = convert_step_to_usd(
        str(input_path),
        str(output_path)
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()