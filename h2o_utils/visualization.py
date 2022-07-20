import cv2
import mano
# import matplotlib
# matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np


def visualize(sample):
    image = sample[0][0].cpu().numpy()
    image = image.transpose(1, 2, 0) * 255
    image = np.ascontiguousarray(image, np.uint8)
    mesh3d = sample[4][0].cpu().numpy()
    boxes = sample[5][0].cpu().numpy()

    fig = plt.figure(figsize=(25, 15))

    width = 2
    height = 1

    # Visualize Bounding Boxes
    ax = fig.add_subplot(height, width, 1)
    ax.title.set_text('Bounding Boxes')
    bb_image = np.copy(image)
    for bb in boxes:
        bb_image = draw_bb(bb_image, bb, [229, 255, 204])
    ax.imshow(bb_image)

    # Loading object faces
    obj_mesh = read_obj('../datasets/objects/mesh_1000/milk.obj')
    obj_faces = obj_mesh.f
    # Load hand faces
    model_path = '../mano_v1_2/models/'
    right_hand_faces = mano.load(model_path=model_path, is_right=True, num_pca_comps=45).faces
    left_hand_faces = mano.load(model_path=model_path, is_right=False, num_pca_comps=45).faces

    # Visualize 3D Meshes
    ax = fig.add_subplot(height, width, 2, projection="3d")
    plot3dVisualize(ax, mesh3d[:778], left_hand_faces, flip_x=True, isOpenGLCoords=False, c="r")
    plot3dVisualize(ax, mesh3d[778:778 * 2], right_hand_faces, flip_x=True, isOpenGLCoords=False, c="g")
    plot3dVisualize(ax, mesh3d[778 * 2:], obj_faces, flip_x=True, isOpenGLCoords=False, c="b")
    cam_equal_aspect_3d(ax, mesh3d, flip_x=False)
    ax.title.set_text('2-Hand Object Mesh')
    plt.show()


def draw_bb(img, bb, color):
    bb_img = np.copy(img)

    bb = bb.astype(int)
    cv2.rectangle(bb_img, (bb[0], bb[1]), (bb[2], bb[3]), color, 1)
    return bb_img


def plot3dVisualize(ax, m, faces, flip_x=False, c="b", alpha=0.1,
                    camPose=np.eye(4, dtype=np.float32),
                    isOpenGLCoords=False):
    '''
    Create 3D visualization
    :param ax: matplotlib axis
    :param m: mesh
    :param flip_x: flix x axis?
    :param c: mesh color
    :param alpha: transperency
    :param camPose: camera pose
    :param isOpenGLCoords: is mesh in openGL coordinate system?
    :return:
    '''
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if hasattr(m, 'r'):
        verts = np.copy(m.r) * 1000
    elif hasattr(m, 'v'):
        verts = np.copy(m.v) * 1000
    elif isinstance(m, np.ndarray):  # In case of an output of a Mano layer (no need to scale)
        verts = np.copy(m)
    else:
        raise Exception('Unknown Mesh format')
    vertsHomo = np.concatenate([verts, np.ones((verts.shape[0], 1), dtype=np.float32)], axis=1)
    verts = vertsHomo.dot(camPose.T)[:, :3]

    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if isOpenGLCoords:
        verts = verts.dot(coordChangeMat.T)

    ax.view_init(elev=90, azim=-90)
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    if c == "b":
        face_color = (141 / 255, 184 / 255, 226 / 255)
        face_color = np.tile(np.array([[0., 0., 1., 1.]]), [verts.shape[0], 1])
        edge_color = (0 / 255, 0 / 255, 112 / 255)
    elif c == "r":
        face_color = (226 / 255, 141 / 255, 141 / 255)
        face_color = np.tile(np.array([[1., 0., 0., 1.]]), [verts.shape[0], 1])
        edge_color = (112 / 255, 0 / 255, 0 / 255)
    elif c == "viridis":
        face_color = plt.cm.viridis(np.linspace(0, 1, faces.shape[0]))
        edge_color = None
        edge_color = (0 / 255, 0 / 255, 112 / 255)
    elif c == "plasma":
        face_color = plt.cm.plasma(np.linspace(0, 1, faces.shape[0]))
        edge_color = None
        # edge_color = (0 / 255, 0 / 255, 112 / 255)
    else:
        face_color = c
        edge_color = c

    mesh.set_edgecolor(edge_color)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    cam_equal_aspect_3d(ax, verts, flip_x=flip_x)
    # plt.tight_layout()


def read_obj(filename):
    """ Reads the Obj file. Function reused from Matthew Loper's OpenDR package"""

    lines = open(filename).read().split('\n')

    d = {'v': [], 'f': []}

    for line in lines:
        line = line.split()
        if len(line) < 2:
            continue

        key = line[0]
        values = line[1:]

        if key == 'v':
            d['v'].append([np.array([float(v) for v in values[:3]])])
        elif key == 'f':
            spl = [l.split('/') for l in values]
            d['f'].append([np.array([int(l[0]) - 1 for l in spl[:3]], dtype=np.uint32)])

    for k, v in d.items():
        if k in ['v', 'f']:
            if v:
                d[k] = np.vstack(v)
            else:
                print(k)
                del d[k]
        else:
            d[k] = v

    result = Minimal(**d)

    return result


class Minimal(object):
    def __init__(self, **kwargs):
        self.__dict__ = kwargs


def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)
