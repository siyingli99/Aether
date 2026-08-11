import cv2 
import numpy as np
from open3d import geometry
import math
from dataparser import get_intrinsic
from nuscenes.utils.data_classes import Box
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from tools.analysis_tools.visualize.mini_run import Visualizer
import glob
import os
import mmcv
import mediapy as media

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def draw_bev(pred_bboxes, out):
    rot_axis=2
    canvas_h, canvas_w = 400, 200
    zoom = 5
    canvas = np.ones((canvas_h, canvas_w, 3)) * 255
    for i in range(len(pred_bboxes)):
        center = pred_bboxes[i, 0:3]
        dim = pred_bboxes[i, 3:6]
        yaw = np.zeros(3)
        yaw[rot_axis] = -pred_bboxes[i, 6]
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)
        center[rot_axis] += dim[rot_axis] / 2
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)
        box3d_verts = np.asarray(box3d.get_box_points())
        bev_box_verts = box3d_verts[[0, 1, 2, 7], :2] * zoom
        bev_box_verts[:, 0] = -bev_box_verts[:, 0] + canvas_w // 2
        bev_box_verts[:, 1] += canvas_h // 2
        bev_box_verts = bev_box_verts.astype(np.int)
        bev_box_verts = bev_box_verts[:, [1,0]]
        connects = [[0, 1], [1, 3], [3, 2], [2, 0]]
        for conn in connects:
            cv2.line(canvas, bev_box_verts[conn[0]], bev_box_verts[conn[1]], (255, 0, 0), 2)
    cv2.imwrite(out, canvas)

def draw_proj(results, raw_data, parse_data, out, cameras):
    obs, info, cam_params = raw_data
    l2i_mats = parse_data['img_metas'][0][0]['lidar2img']

    box3d  = results['boxes_3d_det']
    scores = results['scores_3d_det'].numpy()
    labels = results['labels_3d_det'].numpy()

    box_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    box_yaw = -box_yaw - np.pi / 2

    imgs = []
    for cam, l2i in zip(cameras, l2i_mats):
        l2c = cam_params[cam]['l2c']
        K = get_intrinsic(cam_params[cam]['intrinsic'])
        im = obs['rgb'][cam]
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        # im = cv2.resize(im, (640, 380))

        for i in range(len(box3d)):
            if scores[i] < 0.25:
                continue
            quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
            velocity = (*box3d.tensor[i, 7:9], 0.0)
            box = Box(
                box_center[i],
                box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity)
            # import ipdb; ipdb.set_trace()
            # box.render_cv2(im, K @ l2c, normalize=True)

            box3d_verts = box.corners().T
            cam_verts = (l2c[:3, :3] @ box3d_verts.T).T + l2c[:3, 3]
            uvz_verts = (K[:3, :3] @ cam_verts.T).T
            # uvz_verts = (l2i[:3, :3] @ box3d_verts.T).T + l2i[:3, 3]
            uv_verts = uvz_verts[:, :2] / uvz_verts[:, 2][:, None]

            mask = (uvz_verts[:, 2] > 0) & (uv_verts[:, 0] >= 0) & (uv_verts[:, 1] >= 0) & (uv_verts[:, 1] < im.shape[0]) & (uv_verts[:, 0] < im.shape[1])
            if mask.sum() < 4:
                continue

            connections = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                            [0, 4], [1, 5], [2, 6], [3, 7]]

            for connection in connections:
                line = uv_verts[connection, :].astype(int).tolist()
                cv2.line(im, line[0], line[1], color=(0,0,255), thickness=1)

        imgs.append(im)

    cat_images = cv2.vconcat([
        cv2.hconcat([imgs[2], imgs[0], imgs[1]]),
        cv2.hconcat([imgs[5], imgs[3], imgs[4]]),
    ])
    
    cv2.imwrite(out, cat_images)
    
def get_bev_img(outputs, save_fn):
    render_cfg = dict(
        with_occ_map=False,
        with_map=False,
        with_planning=False,
        with_pred_box=True,
        with_pred_traj=True,
        show_command=False,
        show_sdc_car=True,
        show_legend=True,
        show_sdc_traj=False
    )
    viser = Visualizer(**render_cfg)
    prediction_dict = viser._parse_predictions(outputs)
    viser.bev_render.reset_canvas(dx=1, dy=1)
    viser.bev_render.set_plot_cfg()

    viser.bev_render.render_pred_box_data(
            prediction_dict['predicted_agent_list'])
    viser.bev_render.render_pred_traj(
            prediction_dict['predicted_agent_list'])
    # viser.bev_render.render_pred_map_data(
    #         prediction_dict['predicted_map_seg'])
    # viser.bev_render.render_pred_box_data(
    #     [prediction_dict['predicted_planning']])
    # viser.bev_render.render_planning_data(
    #     prediction_dict['predicted_planning'], show_command=viser.show_command)
    viser.bev_render.render_sdc_car()
    viser.bev_render.render_legend()
    
    # convert img to byteIO
    # img_buffer = io.BytesIO()
    viser.bev_render.save_fig(save_fn)
    # img_buffer.seek(0)
    
    # # convert byteIO to base64
    # img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    # return img_base64

def save_visualize_img(outputs, visual_imgs, save_fn):
    #visiualize
    render_cfg = dict(
        with_occ_map=False,
        with_map=True,
        with_planning=True,
        with_pred_box=True,
        with_pred_traj=True,
        show_command=True,
        show_sdc_car=True,
        show_legend=True,
        show_sdc_traj=False
    )
    CAM_NAMES = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ]

    viser = Visualizer(**render_cfg)
    prediction_dict = viser._parse_predictions(outputs)
    metas = mmcv.load('data_temp/nus_vis_pose_info.pkl')
    # breakpoint()

    image_dict = {} # TODO: surrounding images
    if visual_imgs is not None:
        visual_imgs = np.array(visual_imgs, dtype=np.float32)
        visual_imgs /= 255.
        temp=np.zeros((6,4,1600,3))
        visual_imgs = np.concatenate([visual_imgs,temp],1)
        image_dict['CAM_FRONT_LEFT'] = visual_imgs[2]
        image_dict['CAM_FRONT'] = visual_imgs[0]
        image_dict['CAM_FRONT_RIGHT'] = visual_imgs[1]
        image_dict['CAM_BACK_RIGHT'] = visual_imgs[5]
        image_dict['CAM_BACK'] = visual_imgs[3]
        image_dict['CAM_BACK_LEFT'] = visual_imgs[4]
    else:
        for cam in CAM_NAMES:
            image_dict[cam] = np.zeros((900, 1600, 3)) # TODO: replace the images
    sample_info = {}
    sample_info['images'] = {}
    sample_info['metas'] = metas
    sample_info['images'] = image_dict
    '''
    sample_info:
        - 'images': 
            'CAM_FRONT': np.array
        - 'metas': 
            'lidar_cs_record'
            'CAM_FRONT':
                'cs_record'
                'imsize'
                'cam_intrinsic'
    }
    '''

    viser.visualize_bev(prediction_dict, save_fn)
    viser.visualize_cam(prediction_dict, sample_info, save_fn)
    viser.combine(save_fn) 
    
def to_video(folder_path, fps=5, downsample=1):
    imgs_path = glob.glob(os.path.join(folder_path, '*_combine.jpg'))
    imgs_path = sorted(imgs_path)
    img_array = []
    for img_path in imgs_path:
        img = cv2.imread(img_path)
        height, width, channel = img.shape
        img = cv2.resize(img, (width//downsample, height//downsample))
        height, width, channel = img.shape
        size = (width, height)
        img_array.append(img)
        
    # media.write_video(os.path.join(folder_path, 'video.mp4'), img_array, fps=10)
    mp4_path = os.path.join(folder_path, 'video.mp4')
    if os.path.exists(mp4_path): 
        os.remove(mp4_path)
    out = cv2.VideoWriter(
        mp4_path, 
        cv2.VideoWriter_fourcc(*'DIVX'), fps, size
    )
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()