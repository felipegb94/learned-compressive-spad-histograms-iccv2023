import cv2
import scipy.io
import open3d as o3d
import tof_utils
import numpy as np

BASE = "/nobackup/bhavya/votenet/sunrgbd/sunrgbd_trainval/"

scenes = open(BASE + 'val_data_idx.txt').readlines()
scenes = [x.strip() for x in scenes]

GEN_FOLDER = 'processed_full/SimSPADDataset_nr-576_nc-704_nt-1024_tres-586ps_dark-0_psf-0/'
for scene in scenes:
    data = scipy.io.loadmat(BASE + GEN_FOLDER + 'spad_' + scene.zfill(6) + '_50_1.mat')

    nr, nc = data['intensity'].shape
    # print(data['intensity'].shape)
    nt = data['num_bins'][0,0]
    K = data['K']
    print(K)
    range_bins = data['range_bins']
    dist = tof_utils.tof2depth(range_bins*data['bin_size'])

    # spad = data['spad'].toarray()
    # spad = spad.reshape((nr, nc, nt), order='F')
    # spad = spad.argmax(-1)
    # dist = tof_utils.tof2depth(spad*data['bin_size'])
    # mindist = dist.copy()
    # mindist = mindist.reshape(-1)
    # mindist.sort()
    # print(mindist[:100])

    # depthmap = dist

    cx, cy = K[1, 6], K[1, 7]
    fx, fy = K[1, 0], K[1, 4]
    xx = np.linspace(1, nc, nc)
    yy = np.linspace(1, nr, nr)
    x, y = np.meshgrid(xx, yy)
    print(cx, cy, fx, fy)
    x = (x - cx)/fx
    y = (y - cy)/fy
    depthmap = dist/(x**2 + y**2 + 1)**0.5

    # pts = []
    # for ii in range(1, nr):
    #     for jj in range(1, nc):
    #         x, y = jj-cx, ii-cy
    #         dd = dist[ii-1,jj-1]
    #         depth = dd / ( ( (x/fx)**2 + (y/fy)**2 + 1 )**0.5 )
    #         X = x*depth/fx
    #         Y = y*depth/fy
    #         depthmap[ii-1,jj-1] = depth
            # pts.append((X, depth, -Y))

    depthmap = depthmap*1000.
    depthmap = depthmap.astype(np.uint16)
    # Not sure why SUNRGBD code for converting to point cloud (read3dPoints.m) shifts last 3 bits, but I am zeroing it out for now
    depthmap = (depthmap>>3)<<3
    cv2.imwrite(BASE + GEN_FOLDER + 'spad_' + scene.zfill(6) + '_50_1_gtdepth.png', depthmap)
    # cv2.imwrite('intensity.jpg', np.repeat(data['intensity'][:, :, np.newaxis], 3, axis=2)*255 )

    # pts = np.array(pts)
    # pts = pts[np.random.choice(pts.shape[0], 100000, replace=False), :]
    # # print(pts[:10])
    # write_text = True
    # use_o3d(pts, write_text)










def use_o3d(pts, write_text):
    pcd = o3d.geometry.PointCloud()

    # the method Vector3dVector() will convert numpy array of shape (n, 3) to Open3D format.
    # see http://www.open3d.org/docs/release/python_api/open3d.utility.Vector3dVector.html#open3d.utility.Vector3dVector
    pcd.points = o3d.utility.Vector3dVector(pts)

    # http://www.open3d.org/docs/release/python_api/open3d.io.write_point_cloud.html#open3d.io.write_point_cloud
    o3d.io.write_point_cloud("my_pts.ply", pcd, write_ascii=write_text)

    # read ply file
    pcd = o3d.io.read_point_cloud('my_pts.ply')

    # visualize
    o3d.visualization.draw_geometries([pcd])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D