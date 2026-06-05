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
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# GT pose helper
# ---------------------------------------------------------------------------

def get_gt_pose_for_frame(scene_gt: dict, frame_idx: int, ob_id: int):
    """
    Return a 4x4 pose (float64, metres) for the given frame/object, or None.
    cam_t_m2c is in millimetres in the JSON; divided by 1000 here.
    """
    frame_key = str(frame_idx)
    if frame_key not in scene_gt:
        return None
    for entry in scene_gt[frame_key]:
        if entry['obj_id'] == ob_id:
            R = np.array(entry['cam_R_m2c'], dtype=np.float64).reshape(3, 3)
            t = np.array(entry['cam_t_m2c'], dtype=np.float64) * 1e-3
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = R
            pose[:3,  3] = t
            return pose
    return None


def get_all_ob_ids_in_frame(scene_gt: dict, frame_idx: int):
    """Return list of all obj_ids present in a given frame."""
    frame_key = str(frame_idx)
    if frame_key not in scene_gt:
        return []
    return [entry['obj_id'] for entry in scene_gt[frame_key]]


# ---------------------------------------------------------------------------
# GT-seeded initialisation
# ---------------------------------------------------------------------------

def perturb_pose(R_gt: np.ndarray, t_gt: np.ndarray,
                 rot_noise_deg: float = 10.0,
                 trans_noise_m: float = 0.01) -> np.ndarray:
    """
    Return a 4x4 pose within rot_noise_deg / trans_noise_m of GT.
    t_gt and trans_noise_m must both be in METRES.
    """
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle_rad = np.deg2rad(np.random.uniform(-rot_noise_deg, rot_noise_deg))
    R_noise = Rotation.from_rotvec(axis * angle_rad).as_matrix()
    R_init  = R_noise @ R_gt
    t_init  = t_gt + np.random.uniform(-trans_noise_m, trans_noise_m, size=3)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R_init
    pose[:3,  3] = t_init
    return pose


# ---------------------------------------------------------------------------
# Error metric helpers
# ---------------------------------------------------------------------------

def translation_error(pose_est: np.ndarray, pose_gt: np.ndarray) -> float:
    return float(np.linalg.norm(pose_est[:3, 3] - pose_gt[:3, 3]))


def rotation_error_deg(pose_est: np.ndarray, pose_gt: np.ndarray) -> float:
    R_rel = pose_est[:3, :3].T @ pose_gt[:3, :3]
    cos_angle = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def add_error(pose_est: np.ndarray, pose_gt: np.ndarray,
              model_pts: np.ndarray) -> float:
    pts_h   = np.hstack([model_pts, np.ones((len(model_pts), 1))])
    pts_est = (pose_est @ pts_h.T).T[:, :3]
    pts_gt  = (pose_gt  @ pts_h.T).T[:, :3]
    return float(np.mean(np.linalg.norm(pts_est - pts_gt, axis=1)))


def pose_to_flat(pose: np.ndarray):
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
    parser.add_argument('--debug',             type=int, default=1)
    parser.add_argument('--debug_dir',         type=str, default=f'{code_dir}/debug')
    parser.add_argument('--num_frames',        type=int, default=0,
                        help='Frames to process. 0 = all available.')

    # ob_id selection: single id or all
    parser.add_argument('--ob_id', type=int, default=-1,
                        help='Single obj_id to track. -1 = all objects in scene_gt (default).')

    # GT-seeded initialisation flags (mirror pose_estimation_dr.py)
    parser.add_argument('--use_gt_init', action='store_true',
                        help='Seed every frame pose from GT + small noise instead of '
                             'running the full rotation-grid search. Mirrors '
                             'USE_GT_INITIALIZATION in pose_estimation_dr.py.')
    parser.add_argument('--use_gt_init_only', action='store_true',
                        help='Like --use_gt_init but skip the refiner entirely and '
                             'use the perturbed GT pose directly as the estimate. '
                             'Mirrors USE_GT_INIT_ONLY in pose_estimation_dr.py. '
                             'Applied to EVERY frame (same as DR script behaviour).')
    parser.add_argument('--init_rot_noise_deg',  type=float, default=10.0,
                        help='Rotation noise on GT seed (degrees). Default 10.')
    parser.add_argument('--init_trans_noise_mm', type=float, default=10.0,
                        help='Translation noise on GT seed (mm). Default 10.')
    args = parser.parse_args()

    if args.use_gt_init_only:
        args.use_gt_init = True

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

    # Note: FoundationPose is re-created per object when tracking multiple
    # objects, so this initial construction is just for single-ob_id mode.
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

    # Subsample mesh vertices for ADD
    model_pts = mesh.vertices
    if len(model_pts) > 1000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(model_pts), 1000, replace=False)
        model_pts = model_pts[idx]

    # ------------------------------------------------------------------
    # Determine which object IDs to process
    # ------------------------------------------------------------------
    if reader.scene_gt is None:
        logging.error("scene_gt not loaded by reader — cannot determine object IDs.")
        raise RuntimeError("scene_gt required")

    if args.ob_id >= 0:
        ob_ids_to_track = [args.ob_id]
    else:
        # Collect all unique obj_ids that appear across all frames
        ob_ids_to_track = sorted({
            entry['obj_id']
            for fdata in reader.scene_gt.values()
            for entry in fdata
        })
    logging.info(f"Object IDs to track: {ob_ids_to_track}")

    # ------------------------------------------------------------------
    # CSV setup — add obj_id column to match DR script
    # ------------------------------------------------------------------
    csv_path   = os.path.join(debug_dir, 'pose_errors.csv')
    est_cols   = [f'est_pose_{r}{c}' for r in range(4) for c in range(4)]
    gt_cols    = [f'gt_pose_{r}{c}'  for r in range(4) for c in range(4)]
    fieldnames = (['frame_id', 'obj_id'] + est_cols + gt_cols
                  + ['translation_error_m', 'rotation_error_deg', 'add_error_m'])

    csv_file   = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    logging.info(f"CSV will be written to: {csv_path}")

    # ------------------------------------------------------------------
    # Main loop — outer: object, inner: frame
    #
    # Why this structure?
    # -------------------
    # FoundationPose maintains internal state (pose_last) across frames for
    # a single object.  To track multiple objects we run the full frame
    # sequence once per object, keeping a separate FoundationPose estimator
    # per object.
    #
    # When --use_gt_init_only is active (matching DR script behaviour) the
    # estimator is re-seeded from GT every frame anyway, so the loop order
    # doesn't matter for correctness — but per-object outer loop is cleaner.
    # ------------------------------------------------------------------
    num_frames = (min(args.num_frames, len(reader.color_files))
                  if args.num_frames > 0 else len(reader.color_files))

    for ob_id in ob_ids_to_track:
        logging.info(f"=== Processing obj_id={ob_id} ===")

        # Fresh estimator per object so pose_last doesn't bleed between objects
        est_obj = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=debug_dir,
            debug=debug,
            glctx=glctx,
        )

        for i in range(num_frames):
            logging.info(f'obj_id={ob_id}  frame={i}')

            frame_key_padded = reader.id_strs[i]
            if frame_key_padded in reader.K_table:
                reader.K = reader.K_table[frame_key_padded]

            color = reader.get_color(i).astype(np.uint8)
            depth = reader.get_depth(i).astype(np.float32)

            # GT pose for this frame + object
            gt_pose = get_gt_pose_for_frame(reader.scene_gt, i, ob_id)
            if gt_pose is None:
                logging.warning(f"No GT for frame {i}, obj_id {ob_id} — skipping.")
                continue

            # ----------------------------------------------------------
            # Pose estimation
            #
            # Modes:
            #
            #  --use_gt_init_only  (matches DR USE_GT_INIT_ONLY=True)
            #      Re-seed from perturbed GT every frame, skip refiner.
            #      Guarantees error <= noise bounds on every frame.
            #      Use this to make FoundationPose directly comparable to
            #      the DR script results.
            #
            #  --use_gt_init  (without _only)
            #      Re-seed from perturbed GT every frame, then refine with
            #      track_one.  Better than blind register() but refiner may
            #      still drift on thin/occluded tools.
            #
            #  (default, neither flag)
            #      Frame 0: register() — full rotation-grid search.
            #      Frame 1+: track_one() — tracks from previous frame.
            #      This is the original FoundationPose pipeline.
            # ----------------------------------------------------------

            if args.use_gt_init:
                trans_noise_m = args.init_trans_noise_mm * 1e-3
                init_pose = perturb_pose(
                    gt_pose[:3, :3], gt_pose[:3, 3],
                    rot_noise_deg=args.init_rot_noise_deg,
                    trans_noise_m=trans_noise_m,
                )

                if args.use_gt_init_only:
                    # Use perturbed GT directly — no refiner
                    pose = init_pose.astype(np.float32)
                    # Keep pose_last in sync for the (unused) tracker state
                    tf_to_center = est_obj.get_tf_to_centered_mesh().data.cpu().numpy()
                    pose_centered = init_pose @ np.linalg.inv(tf_to_center)
                    est_obj.pose_last = torch.tensor(
                        pose_centered, dtype=torch.float32, device='cuda')

                else:
                    # Inject GT-seeded init, then refine
                    tf_to_center = est_obj.get_tf_to_centered_mesh().data.cpu().numpy()
                    pose_centered = init_pose @ np.linalg.inv(tf_to_center)
                    est_obj.pose_last = torch.tensor(
                        pose_centered, dtype=torch.float32, device='cuda')
                    pose = est_obj.track_one(rgb=color, depth=depth, K=reader.K,
                                             iteration=args.est_refine_iter)

            else:
                # Original pipeline: register on frame 0, track_one after
                if i == 0:
                    mask = reader.get_mask(0).astype(bool)
                    torch.cuda.empty_cache()
                    import gc; gc.collect()
                    pose = est_obj.register(K=reader.K, rgb=color, depth=depth,
                                            ob_mask=mask,
                                            iteration=args.est_refine_iter)
                else:
                    pose = est_obj.track_one(rgb=color, depth=depth, K=reader.K,
                                             iteration=args.track_refine_iter)

            # Save raw estimated pose
            os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
            np.savetxt(
                f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}_obj{ob_id}.txt',
                pose.reshape(4, 4))

            # Compute errors
            pose_44 = pose.reshape(4, 4)
            trans_err = translation_error(pose_44, gt_pose)
            rot_err   = rotation_error_deg(pose_44, gt_pose)
            add_val   = add_error(pose_44, gt_pose, model_pts)

            # Write CSV row
            row = {'frame_id': reader.id_strs[i], 'obj_id': ob_id}
            for col, val in zip(est_cols, pose_to_flat(pose_44)):
                row[col] = val
            for col, val in zip(gt_cols, pose_to_flat(gt_pose)):
                row[col] = val
            row['translation_error_m'] = trans_err
            row['rotation_error_deg']  = rot_err
            row['add_error_m']         = add_val
            csv_writer.writerow(row)
            csv_file.flush()

            logging.info(
                f"  obj={ob_id} frame={i}: "
                f"rot={rot_err:.2f} deg  trans={trans_err*1e3:.2f} mm  ADD={add_val*1e3:.2f} mm")

            # Debug visualisation (uses obj 0 estimator for vis; ok for debug)
            if debug >= 1:
                center_pose = pose_44 @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(reader.K, img=color,
                                        ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1,
                                    K=reader.K, thickness=3, transparency=0,
                                    is_input_rgb=True)
                try:
                    cv2.imshow('1', vis[..., ::-1])
                    cv2.waitKey(1)
                except cv2.error:
                    pass  # headless server

            if debug >= 2:
                os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
                imageio.imwrite(
                    f'{debug_dir}/track_vis/{reader.id_strs[i]}_obj{ob_id}.png', vis)

    csv_file.close()
    logging.info(f"Done. Pose error CSV saved to: {csv_path}")
