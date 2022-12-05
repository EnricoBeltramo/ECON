# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import numpy as np
import cv2
import pymeshlab
import torch
import torchvision
import trimesh
import os
from termcolor import colored
import os.path as osp
import _pickle as cPickle

from pytorch3d.structures import Meshes
import torch.nn.functional as F
from lib.common.imutils import uncrop
from lib.common.render_utils import Pytorch3dRasterizer
from pytorch3d.renderer.mesh import rasterize_meshes
from PIL import Image, ImageFont, ImageDraw
from kaolin.ops.mesh import check_sign

from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
import tinyobjloader


def create_grid_points_from_xyz_bounds(bound, res):

    min_x, max_x, min_y, max_y, min_z, max_z = bound
    x = torch.linspace(min_x, max_x, res)
    y = torch.linspace(min_y, max_y, res)
    z = torch.linspace(min_z, max_z, res)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    return torch.stack([X, Y, Z], dim=-1)


def create_grid_points_from_xy_bounds(bound, res):

    min_x, max_x, min_y, max_y = bound
    x = torch.linspace(min_x, max_x, res)
    y = torch.linspace(min_y, max_y, res)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    return torch.stack([X, Y], dim=-1)


def apply_face_mask(mesh, face_mask):

    mesh.update_faces(face_mask)
    mesh.remove_unreferenced_vertices()

    return mesh


def apply_vertex_mask(mesh, vertex_mask):

    faces_mask = vertex_mask[mesh.faces].any(dim=1)
    mesh = apply_face_mask(mesh, faces_mask)

    return mesh


def apply_vertex_face_mask(mesh, vertex_mask, face_mask):

    faces_mask = vertex_mask[mesh.faces].any(dim=1) * torch.tensor(face_mask)
    mesh.update_faces(faces_mask)
    mesh.remove_unreferenced_vertices()

    return mesh


def part_removal(full_mesh, vis_mask, part_mesh, thres, device, clean=True, camera_ray=False):

    # thres: face 3e-2, hand 4e-2

    from lib.dataset.PointFeat import PointFeat
    part_extractor = PointFeat(
        torch.tensor(part_mesh.vertices).unsqueeze(0).to(device),
        torch.tensor(part_mesh.faces).unsqueeze(0).to(device))

    (part_dist, part_cos) = part_extractor.query(
        torch.tensor(full_mesh.vertices).unsqueeze(0).to(device), {"smpl_nsdf": None})

    if camera_ray:
        remove_mask = torch.logical_and(part_dist < thres, part_cos > 0.5)
    else:
        remove_mask = part_dist < thres
    
    if vis_mask is not None:
        BNI_verts_mask = ~(torch.logical_or(remove_mask, vis_mask.to(device))).flatten()
    else:
        BNI_verts_mask = ~(remove_mask).flatten()

    BNI_part_mask = BNI_verts_mask[full_mesh.faces].any(dim=1)
    full_mesh.update_faces(BNI_part_mask.detach().cpu())
    full_mesh.remove_unreferenced_vertices()

    if clean:
        full_mesh = clean_floats(full_mesh)

    return full_mesh


def cross(triangles):
    """
    Returns the cross product of two edges from input triangles
    Parameters
    --------------
    triangles: (n, 3, 3) float
      Vertices of triangles
    Returns
    --------------
    crosses : (n, 3) float
      Cross product of two edge vectors
    """
    vectors = np.diff(triangles, axis=1)
    crosses = np.cross(vectors[:, 0], vectors[:, 1])
    return crosses


def tri_area(triangles=None, crosses=None, sum=False):
    """
    Calculates the sum area of input triangles
    Parameters
    ----------
    triangles : (n, 3, 3) float
      Vertices of triangles
    crosses : (n, 3) float or None
      As a speedup don't re- compute cross products
    sum : bool
      Return summed area or individual triangle area
    Returns
    ----------
    area : (n,) float or float
      Individual or summed area depending on `sum` argument
    """
    if crosses is None:
        crosses = cross(triangles)
    area = (np.sum(crosses**2, axis=1)**.5) * .5
    if sum:
        return np.sum(area)
    return area


def sample_surface(triangles, count, area=None):
    """
    Sample the surface of a mesh, returning the specified
    number of points
    For individual triangle sampling uses this method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    Parameters
    ---------
    triangles : (n, 3, 3) float
      Vertices of triangles
    count : int
      Number of points to return
    Returns
    ---------
    samples : (count, 3) float
      Points in space on the surface of mesh
    face_index : (count,) int
      Indices of faces for each sampled point
    """

    # len(mesh.faces) float, array of the areas
    # of each face of the mesh
    if area is None:
        area = tri_area(triangles)

    # total area (float)
    area_sum = np.sum(area)
    # cumulative area (len(mesh.faces))
    area_cum = np.cumsum(area)
    face_pick = np.random.random(count) * area_sum
    face_index = np.searchsorted(area_cum, face_pick)

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = triangles[:, 0]
    tri_vectors = triangles[:, 1:].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = np.random.random((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = torch.tensor(sample_vector + tri_origins).float()

    return samples, face_index


def obj_loader(path, with_uv=True):
    # Create reader.
    reader = tinyobjloader.ObjReader()

    # Load .obj(and .mtl) using default configuration
    ret = reader.ParseFromFile(path)

    # note here for wavefront obj, #v might not equal to #vt, same as #vn.
    attrib = reader.GetAttrib()
    v = np.array(attrib.vertices).reshape(-1, 3)
    vt = np.array(attrib.texcoords).reshape(-1, 2)

    shapes = reader.GetShapes()
    tri = shapes[0].mesh.numpy_indices().reshape(-1, 9)
    f_v = tri[:, [0, 3, 6]]
    f_vt = tri[:, [2, 5, 8]]

    if with_uv:
        face_uvs = vt[f_vt].mean(axis=1)  #[m, 2]
        vert_uvs = np.zeros((v.shape[0], 2), dtype=np.float32)  #[n, 2]
        vert_uvs[f_v.reshape(-1)] = vt[f_vt.reshape(-1)]

        return v, f_v, vert_uvs, face_uvs
    else:
        return v, f_v


class HoppeMesh:

    def __init__(self, verts, faces, uvs=None, texture=None):
        """
        The HoppeSDF calculates signed distance towards a predefined oriented point cloud
        http://hhoppe.com/recon.pdf
        For clean and high-resolution pcl data, this is the fastest and accurate approximation of sdf
        """

        # self.device = torch.device("cuda:0")
        mesh = trimesh.Trimesh(verts, faces, process=False, maintains_order=True)
        self.verts = torch.tensor(verts).float()
        self.faces = torch.tensor(faces).long()
        self.vert_normals = torch.tensor(mesh.vertex_normals).float()

        if (uvs is not None) and (texture is not None):
            self.vertex_colors = trimesh.visual.color.uv_to_color(uvs, texture)
            self.face_normals = torch.tensor(mesh.face_normals).float()

    def contains(self, points):

        labels = check_sign(
            self.verts.unsqueeze(0),
            self.faces,
            torch.tensor(points).float().unsqueeze(0),
        )
        return labels.squeeze(0).numpy()

    def get_colors(self, points, faces):
        """
        Get colors of surface points from texture image through 
        barycentric interpolation.
        - points: [n, 3]
        - return: [n, 4] rgba
        """
        triangles = self.verts[faces]  #[n, 3, 3]
        barycentric = trimesh.triangles.points_to_barycentric(triangles, points)  #[n, 3]
        vert_colors = self.vertex_colors[faces]  #[n, 3, 4]
        point_colors = torch.tensor((barycentric[:, :, None] * vert_colors).sum(axis=1)).float()
        return point_colors

    def triangles(self):
        return self.verts[self.faces].numpy()  #[n, 3, 3]


def tensor2variable(tensor, device):
    return tensor.requires_grad_(True).to(device)


class GMoF(torch.nn.Module):

    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return "rho = {}".format(self.rho)

    def forward(self, residual):
        dist = torch.div(residual, residual + self.rho**2)
        return self.rho**2 * dist


def mesh_edge_loss(meshes, target_length: float = 0.0):
    """
    Computes mesh edge length regularization loss averaged across all meshes
    in a batch. Each mesh contributes equally to the final loss, regardless of
    the number of edges per mesh in the batch by weighting each mesh with the
    inverse number of edges. For example, if mesh 3 (out of N) has only E=4
    edges, then the loss for each edge in mesh 3 should be multiplied by 1/E to
    contribute to the final loss.

    Args:
        meshes: Meshes object with a batch of meshes.
        target_length: Resting value for the edge length.

    Returns:
        loss: Average loss across the batch. Returns 0 if meshes contains
        no meshes or all empty meshes.
    """
    if meshes.isempty():
        return torch.tensor([0.0], dtype=torch.float32, device=meshes.device, requires_grad=True)

    N = len(meshes)
    edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    edge_to_mesh_idx = meshes.edges_packed_to_mesh_idx()  # (sum(E_n), )
    num_edges_per_mesh = meshes.num_edges_per_mesh()  # N

    # Determine the weight for each edge based on the number of edges in the
    # mesh it corresponds to.
    # TODO (nikhilar) Find a faster way of computing the weights for each edge
    # as this is currently a bottleneck for meshes with a large number of faces.
    weights = num_edges_per_mesh.gather(0, edge_to_mesh_idx)
    weights = 1.0 / weights.float()

    verts_edges = verts_packed[edges_packed]
    v0, v1 = verts_edges.unbind(1)
    loss = ((v0 - v1).norm(dim=1, p=2) - target_length)**2.0
    loss_vertex = loss * weights
    # loss_outlier = torch.topk(loss, 100)[0].mean()
    # loss_all = (loss_vertex.sum() + loss_outlier.mean()) / N
    loss_all = loss_vertex.sum() / N

    return loss_all


def remesh(obj, obj_path):
    
    obj.export(obj_path)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    # ms.meshing_decimation_quadric_edge_collapse(targetfacenum=100000)
    ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.Percentage(0.5), adaptive=True)
    ms.apply_coord_laplacian_smoothing()
    ms.save_current_mesh(obj_path[:-4]+"_remesh.obj")
    polished_mesh = trimesh.load_mesh(obj_path[:-4]+"_remesh.obj")

    return polished_mesh


def poisson_remesh(obj_path):
    
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=50000)
    ms.apply_coord_laplacian_smoothing()
    ms.save_current_mesh(obj_path)
    polished_mesh = trimesh.load_mesh(obj_path)

    return polished_mesh



def poisson(mesh, obj_path, depth=10):
    
    if osp.exists(obj_path):
        final_mesh = trimesh.load(obj_path)
        return final_mesh

    from pypoisson import poisson_reconstruction
    faces, vertices = poisson_reconstruction(mesh.vertices, mesh.vertex_normals, depth=depth)

    new_meshes = trimesh.Trimesh(vertices, faces)
    new_mesh_lst = new_meshes.split(only_watertight=False)
    comp_num = [new_mesh.vertices.shape[0] for new_mesh in new_mesh_lst]
    final_mesh = new_mesh_lst[comp_num.index(max(comp_num))]
    final_mesh.export(obj_path)
    
    final_mesh = poisson_remesh(obj_path)

    return final_mesh


def get_mask(tensor, dim):

    mask = torch.abs(tensor).sum(dim=dim, keepdims=True) > 0.0
    mask = mask.type_as(tensor)

    return mask


def blend_rgb_norm(norms, data):

    # norms [N, 3, res, res]

    masks = (norms.sum(dim=1) != norms[0, :, 0, 0].sum()).float().unsqueeze(1)
    norm_mask = F.interpolate(torch.cat([norms, masks], dim=1).detach().cpu(),
                              size=data["uncrop_param"]["box_shape"],
                              mode="bilinear",
                              align_corners=False).permute(0, 2, 3, 1).numpy()
    final = data["img_raw"]

    for idx in range(len(norms)):

        norm_pred = (norm_mask[idx, :, :, :3] + 1.0) * 255.0 / 2.0
        mask_pred = np.repeat(norm_mask[idx, :, :, 3:4], 3, axis=-1)

        norm_ori = unwrap(norm_pred, data["uncrop_param"], idx)
        mask_ori = unwrap(mask_pred, data["uncrop_param"], idx)

        final = final * (1.0 - mask_ori) + norm_ori * mask_ori

    return final.astype(np.uint8)


def unwrap(image, uncrop_param, idx):

    img_uncrop = uncrop(
        image,
        uncrop_param["center"][idx],
        uncrop_param["scale"][idx],
        uncrop_param["crop_shape"],
    )

    img_orig = cv2.warpAffine(
        img_uncrop,
        np.linalg.inv(uncrop_param["M"])[:2, :],
        uncrop_param["ori_shape"][::-1],
        flags=cv2.INTER_CUBIC,
    )

    return img_orig


# Losses to smooth / regularize the mesh shape
def update_mesh_shape_prior_losses(mesh, losses):

    # and (b) the edge length of the predicted mesh
    losses["edge"]["value"] = mesh_edge_loss(mesh)
    # mesh normal consistency
    losses["nc"]["value"] = mesh_normal_consistency(mesh)
    # mesh laplacian smoothing
    losses["laplacian"]["value"] = mesh_laplacian_smoothing(mesh, method="uniform")


def rename(old_dict, old_name, new_name):
    new_dict = {}
    for key, value in zip(old_dict.keys(), old_dict.values()):
        new_key = key if key != old_name else new_name
        new_dict[new_key] = old_dict[key]
    return new_dict


def load_checkpoint(model, cfg):

    model_dict = model.state_dict()
    main_dict = {}
    normal_dict = {}

    device = torch.device(f"cuda:{cfg['test_gpus'][0]}")

    if os.path.exists(cfg.resume_path) and cfg.resume_path.endswith("ckpt"):
        main_dict = torch.load(cfg.resume_path, map_location=device)["state_dict"]

        main_dict = {
            k: v
            for k, v in main_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape and ("reconEngine" not in k) and
            ("normal_filter" not in k) and ("voxelization" not in k)
        }
        print(colored(f"Resume MLP weights from {cfg.resume_path}", "green"))

    if os.path.exists(cfg.normal_path) and cfg.normal_path.endswith("ckpt"):
        normal_dict = torch.load(cfg.normal_path, map_location=device)["state_dict"]

        for key in normal_dict.keys():
            normal_dict = rename(normal_dict, key, key.replace("netG", "netG.normal_filter"))

        normal_dict = {
            k: v
            for k, v in normal_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        print(colored(f"Resume normal model from {cfg.normal_path}", "green"))

    model_dict.update(main_dict)
    model_dict.update(normal_dict)
    model.load_state_dict(model_dict)

    model.netG = model.netG.to(device)
    model.reconEngine = model.reconEngine.to(device)

    model.netG.training = False
    model.netG.eval()

    del main_dict
    del normal_dict
    del model_dict

    torch.cuda.empty_cache()

    return model


def read_smpl_constants(folder):
    """Load smpl vertex code"""
    smpl_vtx_std = np.loadtxt(os.path.join(folder, "vertices.txt"))
    min_x = np.min(smpl_vtx_std[:, 0])
    max_x = np.max(smpl_vtx_std[:, 0])
    min_y = np.min(smpl_vtx_std[:, 1])
    max_y = np.max(smpl_vtx_std[:, 1])
    min_z = np.min(smpl_vtx_std[:, 2])
    max_z = np.max(smpl_vtx_std[:, 2])

    smpl_vtx_std[:, 0] = (smpl_vtx_std[:, 0] - min_x) / (max_x - min_x)
    smpl_vtx_std[:, 1] = (smpl_vtx_std[:, 1] - min_y) / (max_y - min_y)
    smpl_vtx_std[:, 2] = (smpl_vtx_std[:, 2] - min_z) / (max_z - min_z)
    smpl_vertex_code = np.float32(np.copy(smpl_vtx_std))
    """Load smpl faces & tetrahedrons"""
    smpl_faces = np.loadtxt(os.path.join(folder, "faces.txt"), dtype=np.int32) - 1
    smpl_face_code = (smpl_vertex_code[smpl_faces[:, 0]] + smpl_vertex_code[smpl_faces[:, 1]] +
                      smpl_vertex_code[smpl_faces[:, 2]]) / 3.0
    smpl_tetras = (np.loadtxt(os.path.join(folder, "tetrahedrons.txt"), dtype=np.int32) - 1)

    return_dict = {
        "smpl_vertex_code": torch.tensor(smpl_vertex_code),
        "smpl_face_code": torch.tensor(smpl_face_code),
        "smpl_faces": torch.tensor(smpl_faces),
        "smpl_tetras": torch.tensor(smpl_tetras)
    }

    return return_dict


def feat_select(feat, select):

    # feat [B, featx2, N]
    # select [B, 1, N]
    # return [B, feat, N]

    dim = feat.shape[1] // 2
    idx = torch.tile(
        (1 - select),
        (1, dim, 1)) * dim + torch.arange(0, dim).unsqueeze(0).unsqueeze(2).type_as(select)
    feat_select = torch.gather(feat, 1, idx.long())

    return feat_select


def get_visibility(xy, z, faces, img_res=2**12, blur_radius=0.0, faces_per_pixel=1):
    """get the visibility of vertices

    Args:
        xy (torch.tensor): [B, N,2]
        z (torch.tensor): [B, N,1]
        faces (torch.tensor): [B, N,3]
        size (int): resolution of rendered image
    """

    if xy.ndimension() == 2:
        xy = xy.unsqueeze(0)
        z = z.unsqueeze(0)
        faces = faces.unsqueeze(0)

    xyz = (torch.cat((xy, -z), dim=-1) + 1.) / 2.
    N_body = xyz.shape[0]
    faces = faces.long().repeat(N_body, 1, 1)
    vis_mask = torch.zeros(size=(N_body, z.shape[1]))
    rasterizer = Pytorch3dRasterizer(image_size=img_res)

    meshes_screen = Meshes(verts=xyz, faces=faces)
    pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
        meshes_screen,
        image_size=rasterizer.raster_settings.image_size,
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        bin_size=rasterizer.raster_settings.bin_size,
        max_faces_per_bin=rasterizer.raster_settings.max_faces_per_bin,
        perspective_correct=rasterizer.raster_settings.perspective_correct,
        cull_backfaces=rasterizer.raster_settings.cull_backfaces,
    )

    pix_to_face = pix_to_face.detach().cpu().view(N_body, -1)
    faces = faces.detach().cpu()

    for idx in range(N_body):
        Num_faces = len(faces[idx])
        vis_vertices_id = torch.unique(
            faces[idx][torch.unique(pix_to_face[idx][pix_to_face[idx] != -1]) - Num_faces * idx, :])
        vis_mask[idx, vis_vertices_id] = 1.0

    # print("------------------------\n")
    # print(f"keep points : {vis_mask.sum()/len(vis_mask)}")

    return vis_mask


def barycentric_coordinates_of_projection(points, vertices):
    """https://github.com/MPI-IS/mesh/blob/master/mesh/geometry/barycentric_coordinates_of_projection.py"""
    """Given a point, gives projected coords of that point to a triangle
    in barycentric coordinates.
    See
        **Heidrich**, Computing the Barycentric Coordinates of a Projected Point, JGT 05
        at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf
    
    :param p: point to project. [B, 3]
    :param v0: first vertex of triangles. [B, 3]
    :returns: barycentric coordinates of ``p``'s projection in triangle defined by ``q``, ``u``, ``v``
            vectorized so ``p``, ``q``, ``u``, ``v`` can all be ``3xN``
    """
    # (p, q, u, v)
    v0, v1, v2 = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    u = v1 - v0
    v = v2 - v0
    n = torch.cross(u, v)
    sb = torch.sum(n * n, dim=1)
    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    sb[sb == 0] = 1e-6
    oneOver4ASquared = 1.0 / sb
    w = points - v0
    b2 = torch.sum(torch.cross(u, w) * n, dim=1) * oneOver4ASquared
    b1 = torch.sum(torch.cross(w, v) * n, dim=1) * oneOver4ASquared
    weights = torch.stack((1 - b1 - b2, b1, b2), dim=-1)
    # check barycenric weights
    # p_n = v0*weights[:,0:1] + v1*weights[:,1:2] + v2*weights[:,2:3]
    return weights


def orthogonal(points, calibrations, transforms=None):
    """
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 3, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    """
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def projection(points, calib):
    if torch.is_tensor(points):
        calib = torch.as_tensor(calib) if not torch.is_tensor(calib) else calib
        return torch.mm(calib[:3, :3], points.T).T + calib[:3, 3]
    else:
        return np.matmul(calib[:3, :3], points.T).T + calib[:3, 3]


def load_calib(calib_path):
    calib_data = np.loadtxt(calib_path, dtype=float)
    extrinsic = calib_data[:4, :4]
    intrinsic = calib_data[4:8, :4]
    calib_mat = np.matmul(intrinsic, extrinsic)
    calib_mat = torch.from_numpy(calib_mat).float()
    return calib_mat


def normalize_v3(arr):
    """ Normalize a numpy array of 3 component vectors shape=(n,3) """
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    vert_norms = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    face_norms = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(face_norms)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    vert_norms[faces[:, 0]] += face_norms
    vert_norms[faces[:, 1]] += face_norms
    vert_norms[faces[:, 2]] += face_norms
    normalize_v3(vert_norms)

    return vert_norms, face_norms


def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, vertices.shape[-1]))

    return vertices[faces.long()]


def compute_normal_batch(vertices, faces):

    if faces.shape[0] != vertices.shape[0]:
        faces = faces.repeat(vertices.shape[0], 1, 1)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]

    vert_norm = torch.zeros(bs * nv, 3).type_as(vertices)
    tris = face_vertices(vertices, faces)
    face_norm = F.normalize(
        torch.cross(tris[:, :, 1] - tris[:, :, 0], tris[:, :, 2] - tris[:, :, 0]),
        dim=-1,
    )

    faces = (faces + (torch.arange(bs).type_as(faces) * nv)[:, None, None]).view(-1, 3)

    vert_norm[faces[:, 0]] += face_norm.view(-1, 3)
    vert_norm[faces[:, 1]] += face_norm.view(-1, 3)
    vert_norm[faces[:, 2]] += face_norm.view(-1, 3)

    vert_norm = F.normalize(vert_norm, dim=-1).view(bs, nv, 3)

    return vert_norm


def calculate_mIoU(outputs, labels):

    SMOOTH = 1e-6

    outputs = outputs.int()
    labels = labels.int()

    intersection = ((outputs & labels).float().sum())  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum()  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = (torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
                  )  # This is equal to comparing with thresolds

    return (thresholded.mean().detach().cpu().numpy()
           )  # Or thresholded.mean() if you are interested in average across the batch


def add_alpha(colors, alpha=0.7):

    colors_pad = np.pad(colors, ((0, 0), (0, 1)), mode="constant", constant_values=alpha)

    return colors_pad


def get_optim_grid_image(per_loop_lst, loss=None, nrow=4, type="smpl"):

    font_path = os.path.join(os.path.dirname(__file__), "tbfo.ttf")
    font = ImageFont.truetype(font_path, 30)
    grid_img = torchvision.utils.make_grid(torch.cat(per_loop_lst, dim=0), nrow=nrow, padding=0)
    grid_img = Image.fromarray(
        ((grid_img.permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 0.5 * 255.0).astype(np.uint8))

    if False:
        # add text
        draw = ImageDraw.Draw(grid_img)
        grid_size = 512
        if loss is not None:
            draw.text((10, 5), f"error: {loss:.3f}", (255, 0, 0), font=font)

        if type == "smpl":
            for col_id, col_txt in enumerate([
                    "image",
                    "smpl-norm(render)",
                    "cloth-norm(pred)",
                    "diff-norm",
                    "diff-mask",
            ]):
                draw.text((10 + (col_id * grid_size), 5), col_txt, (255, 0, 0), font=font)
        elif type == "cloth":
            for col_id, col_txt in enumerate(
                ["image", "cloth-norm(recon)", "cloth-norm(pred)", "diff-norm"]):
                draw.text((10 + (col_id * grid_size), 5), col_txt, (255, 0, 0), font=font)
            for col_id, col_txt in enumerate(["0", "90", "180", "270"]):
                draw.text(
                    (10 + (col_id * grid_size), grid_size * 2 + 5),
                    col_txt,
                    (255, 0, 0),
                    font=font,
                )
        else:
            print(f"{type} should be 'smpl' or 'cloth'")

    grid_img = grid_img.resize((grid_img.size[0], grid_img.size[1]), Image.ANTIALIAS)

    return grid_img


def clean_mesh(verts, faces):

    device = verts.device

    mesh_lst = trimesh.Trimesh(verts.detach().cpu().numpy(), faces.detach().cpu().numpy())
    mesh_lst = mesh_lst.split(only_watertight=False)
    comp_num = [mesh.vertices.shape[0] for mesh in mesh_lst]

    mesh_clean = mesh_lst[comp_num.index(max(comp_num))]
    final_verts = torch.as_tensor(mesh_clean.vertices).float().to(device)
    final_faces = torch.as_tensor(mesh_clean.faces).long().to(device)

    return final_verts, final_faces


def clean_floats(mesh):
    thres = mesh.vertices.shape[0] * 1e-2
    mesh_lst = mesh.split(only_watertight=False)
    clean_mesh_lst = [mesh for mesh in mesh_lst if mesh.vertices.shape[0] > thres]
    return sum(clean_mesh_lst)


def mesh_move(mesh_lst, step, scale=1.0):

    trans = np.array([1.0, 0.0, 0.0]) * step

    resize_matrix = trimesh.transformations.scale_and_translate(scale=(scale), translate=trans)

    results = []

    for mesh in mesh_lst:
        mesh.apply_transform(resize_matrix)
        results.append(mesh)

    return results


def rescale_smpl(fitted_path, scale=100, translate=(0, 0, 0)):

    fitted_body = trimesh.load(fitted_path, process=False, maintain_order=True, skip_materials=True)
    resize_matrix = trimesh.transformations.scale_and_translate(scale=(scale), translate=translate)

    fitted_body.apply_transform(resize_matrix)

    return np.array(fitted_body.vertices)


class SMPLX:

    def __init__(self):

        self.current_dir = osp.join(osp.dirname(__file__), "../../data/smpl_related")

        self.smpl_verts_path = osp.join(self.current_dir, "smpl_data/smpl_verts.npy")
        self.smpl_faces_path = osp.join(self.current_dir, "smpl_data/smpl_faces.npy")
        self.smplx_verts_path = osp.join(self.current_dir, "smpl_data/smplx_verts.npy")
        self.smplx_faces_path = osp.join(self.current_dir, "smpl_data/smplx_faces.npy")
        self.cmap_vert_path = osp.join(self.current_dir, "smpl_data/smplx_cmap.npy")

        self.smplx_to_smplx_path = osp.join(self.current_dir, "smpl_data/smplx_to_smpl.pkl")

        self.smplx_eyeball_fid_path = osp.join(self.current_dir, "smpl_data/eyeball_fid.npy")
        self.smplx_fill_mouth_fid_path = osp.join(self.current_dir, "smpl_data/fill_mouth_fid.npy")
        self.smplx_flame_vid_path = osp.join(self.current_dir,
                                             "smpl_data/FLAME_SMPLX_vertex_ids.npy")
        self.smplx_mano_vid_path = osp.join(self.current_dir, "smpl_data/MANO_SMPLX_vertex_ids.pkl")
        self.front_flame_path = osp.join(self.current_dir, "smpl_data/FLAME_face_mask_ids.npy")

        self.smplx_faces = np.load(self.smplx_faces_path)
        self.smplx_verts = np.load(self.smplx_verts_path)
        self.smpl_verts = np.load(self.smpl_verts_path)
        self.smpl_faces = np.load(self.smpl_faces_path)

        self.smplx_eyeball_fid_mask = np.load(self.smplx_eyeball_fid_path)
        self.smplx_mouth_fid = np.load(self.smplx_fill_mouth_fid_path)
        self.smplx_mano_vid_dict = np.load(self.smplx_mano_vid_path, allow_pickle=True)
        self.smplx_mano_vid = np.concatenate(
            [self.smplx_mano_vid_dict["left_hand"], self.smplx_mano_vid_dict["right_hand"]])
        self.smplx_flame_vid = np.load(self.smplx_flame_vid_path, allow_pickle=True)
        self.smplx_front_flame_vid = self.smplx_flame_vid[np.load(self.front_flame_path)]

        self.front_flame_vertex_mask = torch.zeros(self.smplx_verts.shape[0],).index_fill_(
            0, torch.tensor(self.smplx_front_flame_vid), 1.0)
        self.mano_vertex_mask = torch.zeros(self.smplx_verts.shape[0],).index_fill_(
            0, torch.tensor(self.smplx_mano_vid), 1.0)
        self.eyeball_vertex_mask = torch.zeros(self.smplx_verts.shape[0],).index_fill_(
            0, torch.tensor(self.smplx_faces[self.smplx_eyeball_fid_mask].flatten()), 1.0)

        self.smplx_to_smpl = cPickle.load(open(self.smplx_to_smplx_path, "rb"))

        self.model_dir = osp.join(self.current_dir, "models")
        self.tedra_dir = osp.join(self.current_dir, "../tedra_data")

        self.ghum_smpl_pairs = torch.tensor([
            (0, 24),
            (2, 26),
            (5, 25),
            (7, 28),
            (8, 27),
            (11, 16),
            (12, 17),
            (13, 18),
            (14, 19),
            (15, 20),
            (16, 21),
            (17, 39),
            (18, 44),
            (19, 36),
            (20, 41),
            (21, 35),
            (22, 40),
            (23, 1),
            (24, 2),
            (25, 4),
            (26, 5),
            (27, 7),
            (28, 8),
            (29, 31),
            (30, 34),
            (31, 29),
            (32, 32),
        ]).long()

        # smpl-smplx correspondence
        self.smpl_joint_ids_24 = np.arange(22).tolist() + [68, 73]
        self.smpl_joint_ids_24_pixie = np.arange(22).tolist() + [61 + 68, 72 + 68]
        self.smpl_joint_ids_45 = (np.arange(22).tolist() + [68, 73] + np.arange(55, 76).tolist())

        self.extra_joint_ids = (np.array([
            61,
            72,
            66,
            69,
            58,
            68,
            57,
            56,
            64,
            59,
            67,
            75,
            70,
            65,
            60,
            61,
            63,
            62,
            76,
            71,
            72,
            74,
            73,
        ]) + 68)

        self.smpl_joint_ids_45_pixie = (np.arange(22).tolist() + self.extra_joint_ids.tolist())

    def cmap_smpl_vids(self, type):

        # smplx_to_smpl.pkl
        # KEYS:
        # closest_faces -   [6890, 3] with smplx vert_idx
        # bc            -   [6890, 3] with barycentric weights

        cmap_smplx = torch.as_tensor(np.load(self.cmap_vert_path)).float()

        if type == "smplx":
            return cmap_smplx

        elif type == "smpl":
            bc = torch.as_tensor(self.smplx_to_smpl["bc"].astype(np.float32))
            closest_faces = self.smplx_to_smpl["closest_faces"].astype(np.int32)
            cmap_smpl = torch.einsum("bij, bi->bj", cmap_smplx[closest_faces], bc)
            return cmap_smpl
