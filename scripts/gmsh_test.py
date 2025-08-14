import gmsh
import sys

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)  # Optional: Print messages to the terminal

# Replace "path/to/your_step_file.step" with the actual path
file_path = "/Users/duck/Documents/Mk11/mk11-livery/test_inputs/Mk10.STEP"

# Load STEP file
gmsh.model.add("my_model")
gmsh.model.occ.importShapes(file_path)
gmsh.model.occ.synchronize()

gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.healShapes()
gmsh.model.occ.synchronize()

# --- Helper: detect if surface is valid for transfinite ---
def surface_is_transfinite(surf_tag):
    # Get all boundary loops of the surface
    loops = gmsh.model.getBoundary([(2, surf_tag)], oriented=False)
    loop_tags = gmsh.model.getBoundary([(2, surf_tag)], oriented=False, recursive=False)
    # If more than one loop → hole detected
    outer_and_inner_loops = gmsh.model.getBoundary([(2, surf_tag)], oriented=False)
    loop_entities = gmsh.model.getEntitiesInBoundingBox(*gmsh.model.getBoundingBox(2, surf_tag), 1)
    
    # Count loops explicitly
    loops_info = gmsh.model.getBoundary([(2, surf_tag)], oriented=False)
    loop_count = gmsh.model.getBoundary([(2, surf_tag)], oriented=False, recursive=True).count

    # Quick method: get loops without recursion
    boundary_loops = gmsh.model.getBoundary([(2, surf_tag)], oriented=False, recursive=False)
    unique_loops = set([ltag for ldim, ltag in boundary_loops])
    has_hole = len(unique_loops) > 1

    # Count unique corner points
    edges = gmsh.model.getBoundary([(2, surf_tag)], oriented=False)
    corner_pts = set()
    for edim, etag in edges:
        pts = gmsh.model.getBoundary([(edim, etag)], oriented=False)
        for pdim, ptag in pts:
            corner_pts.add(ptag)
    num_corners = len(corner_pts)

    return (not has_hole) and (num_corners in (3, 4))

# --- Apply transfinite to curves ---
curves = gmsh.model.getEntities(dim=1)
for dim, tag in curves:
    gmsh.model.mesh.setTransfiniteCurve(tag, 10)  # divisions per edge

# --- Process surfaces ---
transfinite_surfaces = set()
surfaces = gmsh.model.getEntities(dim=2)
for dim, tag in surfaces:
    if surface_is_transfinite(tag):
        gmsh.model.mesh.setTransfiniteSurface(tag)
        gmsh.model.mesh.setRecombine(dim, tag)
        transfinite_surfaces.add(tag)
    else:
        print(f"Surface {tag} skipped (hole or >4 corners)")
        gmsh.model.mesh.setRecombine(dim, tag)

# --- Process volumes ---
volumes = gmsh.model.getEntities(dim=3)
for dim, tag in volumes:
    faces = gmsh.model.getBoundary([(dim, tag)], oriented=False)
    face_tags = [ftag for fdim, ftag in faces if fdim == 2]
    if all(f in transfinite_surfaces for f in face_tags):
        gmsh.model.mesh.setTransfiniteVolume(tag)
        gmsh.model.mesh.setRecombine(dim, tag)
    else:
        print(f"Volume {tag} skipped for transfinite (invalid faces present)")

# --- Mesh ---
gmsh.model.mesh.generate(3)
gmsh.write("structured_selective.obj")

if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()