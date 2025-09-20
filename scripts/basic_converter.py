import sys
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.Interface import Interface_Static
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Extend.DataExchange import write_stl_file

def step_to_stl(step_filename, stl_filename, linear_deflection=0.1, angular_deflection=0.5, parallel=True):
    # Load the STEP file
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_filename)

    if status != IFSelect_RetDone:
        print("Error: Cannot read STEP file.")
        return False

    step_reader.TransferRoots()
    shape = step_reader.OneShape()

    # Mesh the shape
    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, parallel)
    mesh.Perform()

    # Write to STL
    stl_writer = StlAPI_Writer()
    stl_writer.SetASCIIMode(False)  # Set to True if you want ASCII STL
    stl_writer.Write(shape, stl_filename)

    print(f"Successfully converted '{step_filename}' to '{stl_filename}'")
    return True

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python step_to_stl.py input.step output.stl")
    else:
        step_to_stl(sys.argv[1], sys.argv[2])
