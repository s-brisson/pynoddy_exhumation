import pyvista as pv
import pyvistaqt as pvqt
import PVGeo
import numpy as np

def Block3D(result_array, type):
    e = result_array[...,3]
    s = result_array[...,4]
    x = result_array[...,0]
    y = result_array[...,1]
    z = result_array[...,2]

    p = pvqt.BackgroundPlotter()

    pc = pv.PolyData(np.c_[x,y,z])

    if type in ['exhumation']:
        pc['Exhumation'] = e 
    else:
        pc['Standard Deviation'] = s

    spacing = lambda arr: np.unique(np.diff(np.unique(arr)))
    voxelsize = spacing(pc.points[:,0]), spacing(pc.points[:,1]), spacing(pc.points[:,2])

    pc = pc.cast_to_unstructured_grid()

    grid = PVGeo.filters.VoxelizePoints(dx = voxelsize[0][0], dy = voxelsize[1][0], dz = voxelsize[2][0], 
                                        estimate = False).apply(pc)

    p.add_mesh(grid, opacity = 0.5, show_edges = False, lighting = False, cmap = 'viridis')

    p.set_scale(zscale = 1)
    p.camera_position = (320, 200, 2)
    p.show_grid(xlabel = 'X', ylabel = 'Y', zlabel = 'Z')

    p.show()