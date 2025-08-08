import argparse
import sys
import os
from pathlib import Path

# Add the parent directory to Python path to import step_convert
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from bruin_livery.project_structure.project_structure import ProjectStructureManager
from bruin_livery.project_structure import config

def setup_config(args):
    """Update configuration based on command line arguments."""
    if args.linear_deflection is not None:
        config.MESH_LINEAR_DEFLECTION = args.linear_deflection


def main():
    parser = argparse.ArgumentParser()
    
    #required
    parser.add_argument(
        "input_directory",
        type=str,
        help=f"Directory to initialize"
    )

    parser.add_argument(
        "-t", "--template_yml",
        type=str,
        dest="template_yml",
        help=f"Template to based directory off of"
    )
    
    args = parser.parse_args()
    
    manager = ProjectStructureManager(
        template_yml_path=Path("../assets/project_structure.yml"),
        active_project_path=Path(args.input_directory)
    )

    # Scaffold a new or existing project directory from the template safely
    manager.scaffold(Path(args.input_directory), dry_run=True)  # Preview only
    manager.scaffold(Path(args.input_directory), dry_run=False) # Actually scaffold


if __name__ == "__main__":
    main()