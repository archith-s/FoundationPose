# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from estimater import *
from datareader import *
import argparse
import trimesh
import csv
import numpy as np


# ---------------------------------------------------------------------------
# GT pose helper
# ---------------------------------------------------------------------------

def get_gt_pose_for_frame(scene_gt: dict, frame_idx: int, ob_id: int):
    """
    Parse scene_gt.json and return a 4x4 pose matrix (float64, meters) for
    the requested frame and object id, or None if not found.

    scene_gt keys are plain integer strings ("0", "1", ...).
    cam_t_m2c is stored in MILLIMETRES → divided by 1000 here to get metres,
    matching the metre-scale output of FoundationPose.
    cam_R_m2c is a flat 9-element row-major rotation matrix.
    """
    frame_key = str(frame_idx)          # "0", "1", … — NOT zero-padded
    if frame_key not in scene_gt:
        return None

    for entry in scene_gt[frame_key]:
        if entry['obj_id'] == ob_id:
            R = np.array(entry['cam_R_m2c'], dtype=np.float64).reshape(3, 3)
            t = np.array(entry['cam_t_m2c'], dtype=np.float64) * 1e-3  # mm → m

            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = R
            pose[:3,  3] = t
            return pose

    return None  # ob_id not present in this frame


# ---------------------------------------------------------------------------
# Error metric helpers
# ---------------------------------------------------------------------------

def translation_error(pose_est: np.ndarray, pose_gt: np.ndarray) -> float:
    """Euclidean distance between translation vectors (metres)."""
    t_est = pose_est[:3, 3]
    t_gt  = pose_gt[:3, 3]
    return float(np.linalg.norm(t_est - t_gt))


def rotation_error_deg(pose_est: np.ndarray, pose_gt: np.ndarray) -> float:
    """
    Geodesic rotation error in degrees.
    e = arccos( (trace(R_est^T @ R_gt) - 1) / 2 )
    """
    R_est = pose_est[:3, :3]
    R_gt  = pose_gt[:3, :3]
    R_rel = R_est.T @ R_gt
    cos_angle = (np.trace(R_rel) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def add_error(pose_est: np.ndarray, pose_gt: np.ndarray,
              model_pts: np.ndarray) -> float:
    """
    ADD (Average Distance of model points).
    Mean Euclidean distance between model points transformed by est vs GT pose.
    Both poses and model_pts must be in the same units (metres).
    """
    pts_h   = np.hstack([model_pts, np.ones((len(model_pts), 1))])   # (N, 4)
    pts_est = (pose_est @ pts_h.T).T[:, :3]
    pts_gt  = (pose_gt  @ pts_h.T).T[:, :3]
    return float(np.mean(np.linalg.norm(pts_est - pts_gt, axis=1)))


def pose_to_flat(pose: np.ndarray):
    """Return the 4x4 pose matrix as a flat list of 16 floats, row-major."""
    return pose.reshape(16).tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument('--mesh_file', type=str,
                        default=f'{code_dir}/../../surgical_robotics_challenge/ADF/PSMs/LND_420006/high_res/tool pitch link.OBJ')
    parser.add_argument('--test_scene_dir', type=str,
                        default=f'{code_dir}/../../SIMPLELND_data/camera_0/000001')
    parser.add_argument('--est_refine_iter',   type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug',     type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    # ------------------------------------------------------------------
    # FIX: scene_gt.json uses 0-indexed obj_id (0, 1, 2, 3, 4).
    #      Set this to the obj_id of the tool you are tracking.
    #      Default changed from 1 → 0.
    # ------------------------------------------------------------------
    parser.add_argument('--ob_id', type=int, default=0,
                        help='obj_id to look up in scene_gt.json (0-indexed)')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    # ------------------------------------------------------------------
    # Load mesh
    # ------------------------------------------------------------------
    mesh = trimesh.load(args.mesh_file)
    if isinstance(mesh, trimesh.Scene):
        logging.info("Mesh loaded as Scene, merging into a single Trimesh object...")
        mesh = mesh.dump(concatenate=True)

    debug     = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=debug,
        glctx=glctx,
    )
    logging.info("estimator initialization done")

    reader = CustomRosReader(base_dir=args.test_scene_dir, zfar=np.inf)
    reader.resize = 0.5

    # ------------------------------------------------------------------
    # Subsample mesh vertices for ADD (cap at 1000 pts for speed).
    # mesh.vertices are in metres (trimesh loads OBJ in native units).
    # ------------------------------------------------------------------
    model_pts = mesh.vertices
    if len(model_pts) > 1000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(model_pts), 1000, replace=False)
        model_pts = model_pts[idx]

    # ------------------------------------------------------------------
    # CSV setup
    # ------------------------------------------------------------------
    csv_path = os.path.join(debug_dir, 'pose_errors.csv')

    est_cols = [f'est_pose_{r}{c}' for r in range(4) for c in range(4)]
    gt_cols  = [f'gt_pose_{r}{c}'  for r in range(4) for c in range(4)]

    fieldnames = (
        ['frame_id']
        + est_cols
        + gt_cols
        + ['translation_error_m', 'rotation_error_deg', 'add_error_m']
    )

    csv_file   = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    logging.info(f"CSV will be written to: {csv_path}")

    # ------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------
    for i in range(len(reader.color_files)):
        logging.info(f'i:{i}')

        # ------------------------------------------------------------------
        # FIX: fetch per-frame intrinsics from K_table instead of reusing the
        #      first frame's K for the whole sequence.
        # ------------------------------------------------------------------
        frame_key_padded = reader.id_strs[i]          # e.g. "000000"
        if frame_key_padded in reader.K_table:
            reader.K = reader.K_table[frame_key_padded]

        color = reader.get_color(i).astype(np.uint8)
        depth = reader.get_depth(i).astype(np.float32)

        if i == 0:
            mask = reader.get_mask(0).astype(bool)
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            pose = est.register(K=reader.K, rgb=color, depth=depth,
                                ob_mask=mask, iteration=args.est_refine_iter)

            if debug >= 3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, reader.K)
                valid = depth >= 0.001
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=reader.K,
                                 iteration=args.track_refine_iter)

        # Save raw estimated pose
        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4, 4))

        # --------------------------------------------------------------
        # FIX: GT pose lookup
        #   - Use plain integer frame index as key ("0", "1", …), not
        #     zero-padded id_str ("000000") which never matches gt keys.
        #   - Build the 4x4 ourselves so we control the mm→m conversion.
        #   - Do NOT call reader.get_gt_poses() which may not exist or
        #     may return poses still in millimetres.
        # --------------------------------------------------------------
        gt_pose = None
        if reader.scene_gt is not None:
            gt_pose = get_gt_pose_for_frame(reader.scene_gt, i, args.ob_id)
            if gt_pose is None:
                logging.warning(
                    f"No GT pose found for frame {i}, obj_id {args.ob_id} "
                    f"— error columns will be empty."
                )

        # --------------------------------------------------------------
        # Compute errors and write CSV row
        # --------------------------------------------------------------
        row = {'frame_id': reader.id_strs[i]}

        flat_est = pose_to_flat(pose.reshape(4, 4))
        for col, val in zip(est_cols, flat_est):
            row[col] = val

        if gt_pose is not None:
            flat_gt = pose_to_flat(gt_pose)
            for col, val in zip(gt_cols, flat_gt):
                row[col] = val

            pose_44 = pose.reshape(4, 4)
            row['translation_error_m'] = translation_error(pose_44, gt_pose)
            row['rotation_error_deg']  = rotation_error_deg(pose_44, gt_pose)
            row['add_error_m']         = add_error(pose_44, gt_pose, model_pts)
        else:
            for col in gt_cols:
                row[col] = ''
            row['translation_error_m'] = ''
            row['rotation_error_deg']  = ''
            row['add_error_m']         = ''

        csv_writer.writerow(row)
        csv_file.flush()   # write incrementally so data isn't lost on crash

        # --------------------------------------------------------------
        # Debug visualisation (unchanged)
        # --------------------------------------------------------------
        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                                thickness=3, transparency=0, is_input_rgb=True)
            #cv2.imshow('1', vis[..., ::-1])
            #cv2.waitKey(1)

        if debug >= 2:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

    csv_file.close()
    logging.info(f"Done. Pose error CSV saved to: {csv_path}")
