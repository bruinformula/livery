import yaml
import os
from pathlib import Path

class ProjectStructureManager:
    active_project_path: Path
    template_yml_path: Path

    active_project: dict
    template_yml: dict

    def __init__(self, template_yml_path: Path, active_project_path: Path):
        self.active_project = self.scan_directory(active_project_path)
        self.template_yml = self.load_template_yml(template_yml_path)
        self.active_project_path = active_project_path.resolve()
        self.template_yml_path = template_yml_path.resolve()

    @staticmethod
    def load_template_yml(path: Path) -> dict:
        if path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(path.read_text())
        else:
            raise ValueError("Unsupported structure file format.")

    @staticmethod
    def scan_directory(path: Path) -> dict:
        result = {}
        for item in sorted(path.iterdir()):
            if item.is_dir():
                result[item.name] = ProjectStructureManager.scan_directory(item)
            elif item.is_file():
                result[item.name] = None
        return result

    ### New methods for safe scaffold ###

    def scaffold(self, base_path: Path, dry_run: bool = False):
        self._scaffold_recursive(base_path, self.template_yml, dry_run)

    def _scaffold_recursive(self, current_path: Path, structure: dict, dry_run: bool):
        for name, substructure in structure.items():
            folder_path = current_path / name
            if dry_run:
                print(f"[DRY RUN] Would create: {folder_path}")
            else:
                os.makedirs(folder_path, exist_ok=True)
            if isinstance(substructure, dict):
                self._scaffold_recursive(folder_path, substructure, dry_run)