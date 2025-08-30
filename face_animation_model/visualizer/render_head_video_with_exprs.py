from face_animation_model.evaluate.eval_root import *
from face_animation_model.utils.fixseed import fixseed
import scipy
from scipy.spatial.transform import Rotation
# from data.hdtf_dataset.debug_data_loader_faster_collate import pose_processing
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
title_font = {'fontname': 'DejaVu Sans', 'size': '16', 'color': 'black', 'weight': 'normal',
              'verticalalignment': 'bottom'}

import torch
import numpy as np
import os, datetime, glob, importlib
import subprocess
from tqdm import tqdm
import cv2
from face_animation_model.visualizer.util_pyrenderer import Facerender
from face_animation_model.visualizer.util_pyrenderer_final_result_version import Facerender as FHQ
from FLAMEModel.FLAME import FLAME
from glob import glob
from skimage import color
import pickle
from face_animation_model.utils.torch_rotation import *
import trimesh

### hdtf metrics
mean = torch.tensor([-0.05855013, -0.0028288, -0.01964026]).float().reshape(1,3)
std = torch.tensor([0.11084834, 0.10225139, 0.08779355]).float().reshape(1,3)
data_min = torch.tensor([-0.5517633,  -0.47712454, -0.44358182]).float().reshape(1,3)
data_max = torch.tensor([0.39624417, 0.5162673,  0.3789773]).float().reshape(1,3)
class render_helper():
    def __init__(self, config = {}, render_type="std"):
        if len(config) == 0:
            config["flame_model_path"] = os.path.join(os.getenv('HOME'),
                                                      "projects/NeuralMotionSynthesis/FLAMEModel",
                                                      "model/generic_model.pkl")
            config["batch_size"] = 1
            config["shape_params"] = 0
            config["expression_params"] = 0
            config["pose_params"] = 0
            config["number_worker"] = 8
            config["use_3D_translation"] = False

        from FLAMEModel.FLAME import FLAME
        self.face_model = FLAME(config)

        # flame expression model 
        if len(config) == 0:
            config["flame_model_path"] = os.path.join(os.getenv('HOME'),
                                                      "projects/NeuralMotionSynthesis/FLAMEModel",
                                                    #   "model/generic_model.pkl")
                                                      "model/female_model.pkl")
                                                    #   "model/male_model.pkl")
            
            config["batch_size"] = 1
            config["shape_params"] = 0
            config["expression_params"] = 100
            config["pose_params"] = 0
            config["number_worker"] = 8
            config["use_3D_translation"] = False
        
        self.expression_face_model = FLAME(config)


        self.cust_trans = np.zeros((1, 3))

        self.render_type = render_type
        if render_type == "paper_quality":
            self.image_size = (800, 800)
            self.face_render = FHQ()
        
        elif render_type == "ratio_800x1200":
            self.image_size = (800, 1200)
            self.face_render = FHQ(img_size=(800, 1200))
            self.cust_trans[0, 2] = 1.35
            self.cust_trans[0, 1] = 0.01
        
        elif render_type == "custom_camera":
            self.image_size = (800, 1200)
        
            # face_render3
            camera_rotation = np.eye(4)
            camera_rotation[:3, :3] = Rotation.from_euler('z', 0, degrees=True).as_matrix() @ Rotation.from_euler('y', 0, degrees=True).as_matrix() @ Rotation.from_euler(
                'x', -15, degrees=True).as_matrix()
            camera_translation = np.eye(4)
            camera_translation[:3, 3] = np.array([0, 0, 1])
            camera_pose3 = camera_rotation @ camera_translation
            self.face_render = FHQ(img_size=self.image_size, camera_pose=camera_pose3)
        
            self.cust_trans[0, 1] = 0.05
            self.cust_trans[0, 2] = 0.20

        elif render_type == "blender":

            os.environ["BLENDER_DEV"] = "GPU"
            self.blend_file = "dev/blender/blender_rendering/render_studio.blend"
            self.blender_script_path = "dev/blender/blender_rendering/render_script.py"
            self.sr = 128
            self.image_size = (512, 512)
            # self.image_size = (1024, 1024)
            self.nf_frames = 10_000

        elif render_type == "blender_800":

            os.environ["BLENDER_DEV"] = "GPU"
            self.blend_file = "dev/blender/blender_rendering/render_studio.blend"
            self.blender_script_path = "dev/blender/blender_rendering/render_script.py"
            self.sr = 200
            self.image_size = (800, 800)
            self.nf_frames = 10_000

        elif render_type == "blender":

            os.environ["BLENDER_DEV"] = "GPU"
            self.blend_file = "dev/blender/blender_rendering/render_studio.blend"
            self.blender_script_path = "dev/blender/blender_rendering/render_script.py"
            self.sr = 128
            self.image_size = (512, 512)
            # self.image_size = (1024, 1024)
            self.nf_frames = 10_000

        else:
            self.image_size = (512, 512)
            from face_animation_model.visualizer.util_pyrenderer import Facerender
            self.face_render = Facerender()

    def dist_to_rgb(self, errors, min_dist=0.0, max_dist=1.0):
        import matplotlib as mpl
        import matplotlib.cm as cm
        norm = mpl.colors.Normalize(vmin=min_dist, vmax=max_dist)
        cmap = cm.get_cmap(name='jet')
        colormapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        return colormapper.to_rgba(errors)[:, 0:3]

    def error_in_mm(self, pred_verts, gt_verts, vertice_dim=15069):
        pred_verts_mm = pred_verts.view(-1, vertice_dim // 3, 3) * 1000.0
        gt_verts_mm = gt_verts.view(-1, vertice_dim // 3, 3) * 1000.0
        diff_in_mm = pred_verts_mm - gt_verts_mm
        dist_in_mm = torch.sqrt(torch.sum(diff_in_mm ** 2, dim=-1))
        return torch.mean(dist_in_mm, dim=0)  # 5023 x 1

    def render_heat_map(self, out_dir, out_seq_name,
                        gt, pred, template, max_error=10):
        out_pred_rendered_images = []
        vid_frames = []
        for i in tqdm(range(pred.shape[0]), desc="Rendering heat map"):
            error = self.error_in_mm(pred[i],
                                     gt[0, i])
            colours = self.dist_to_rgb(error.cpu().numpy(),
                                       0,
                                       max_error)
            pred_frame = self.face_render.render_heat_map(pred[i].cpu().numpy(),
                                                          self.face_model.faces,
                                                          colours)
            out_pred_rendered_images.append(np.copy(pred_frame))

            cv2.putText(pred_frame, f"max {max_error}mm", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            vid_frames.append(pred_frame)


        out_vid_file = os.path.join(out_dir, out_seq_name + "_eval_nf%s.mp4" % len(vid_frames))
        video_file = self.compose_write_video(out_vid_file, vid_frames, self.image_size)

        return video_file, out_pred_rendered_images

    def render_meshes(self, pred_vertices):
        out_pred_rendered_images = []
        for i in tqdm(range(pred_vertices.shape[0]), desc="Rendering expressions"):
            pred_frame = self.render_exprs_to_image(pred_vertices[i])
            out_pred_rendered_images.append(pred_frame)
        return out_pred_rendered_images

    def add_white_bkg(self, image):
        # Create a white background image
        background = np.ones_like(image) * 255  # White background
        background[:, :, 3] = 255
        alpha_channel = image[:, :, 3]
        # Create a 3-channel alpha mask
        alpha_mask = cv2.cvtColor(alpha_channel, cv2.COLOR_GRAY2BGR)
        # Combine the image and the white background using the alpha mask
        result = np.where(alpha_mask != 0, image[:, :, :3], background[:, :, :3])
        result = np.dstack([result, alpha_channel])
        return result [:, :, :3]

    def visualize_meshes(self, out_dir, out_seq_name, audio_file, pred_vertices, desc="expression"):

        if self.render_type == "blender" or self.render_type == "blender_800":
            out_pred_rendered_images = []
            
            ### dump the meshes
            mesh_out_dir = os.path.join(out_dir, "mesh_dump", out_seq_name)
            os.makedirs(mesh_out_dir, exist_ok=True)
            os.system(f"rm -rf {mesh_out_dir}/*")
            for i in tqdm(range(pred_vertices.shape[0]), desc="dumping mesh"):
                outfile = os.path.join(mesh_out_dir,  "%04d.obj"%i)
                self.dump_mesh(pred_vertices[i].reshape(5023, 3), self.expression_face_model.faces, outfile)

            ### render the meshes using the blender
            render_img_dir = os.path.join(out_dir, "img_dump", out_seq_name)
            os.makedirs(render_img_dir, exist_ok=True)
            # render_cmd = f"blender -b {self.blend_file} --python '{self.blender_script_path}' -- -i {mesh_out_dir} -o {render_img_dir} -s {self.sr} -r {self.image_size[0]} -n {self.nf_frames}"
            # render_cmd = f"/home/bthambiraja/blender-4.0.1-linux-x64/blender -b {self.blend_file} --python '{self.blender_script_path}' -- -i {mesh_out_dir} -o {render_img_dir} -s {self.sr} -r {self.image_size[0]} -n {self.nf_frames}"
            blender_fn = os.environ["BLENDER_PATH"]
            render_cmd = f"{blender_fn} -b {self.blend_file} --python '{self.blender_script_path}' -- -i {mesh_out_dir} -o {render_img_dir} -s {self.sr} -r {self.image_size[0]} -n {self.nf_frames}"
            print()
            print(render_cmd)
            os.system(render_cmd)

            ### load the images for renderting
            img_files = sorted(glob(os.path.join(render_img_dir, "*.png")))[:pred_vertices.shape[0]]
            out_pred_rendered_images = [cv2.imread(x, cv2.IMREAD_UNCHANGED) for x in img_files]
            out_pred_rendered_images = [self.add_white_bkg(x) for x in out_pred_rendered_images]

        else:
            out_pred_rendered_images = []
            for i in tqdm(range(pred_vertices.shape[0]), desc=f"Rendering {desc}"):
                pred_frame = self.render_exprs_to_image(pred_vertices[i])

                if "expression" != desc and desc is not None:
                    cv2.putText(pred_frame, desc, (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                cv2.LINE_AA)
                out_pred_rendered_images.append(pred_frame)
            

        # create the video out file
        out_vid_file = os.path.join(out_dir, out_seq_name + "_eval_nf%s.mp4" % len(out_pred_rendered_images))
        video_file = self.compose_write_video(out_vid_file, out_pred_rendered_images, self.image_size)

        if audio_file is not None:
            video_out=video_file.replace(".mp4", "_aud.mp4")
            self.add_audio_to_video(audio_file, video_file, video_out)
            video_file = video_out

        return video_file, out_pred_rendered_images

    def visualize_meshes_with_kf(self, out_dir, out_seq_name, pred_vertices, key_frame = [], kf_string="keyframe"):

        out_pred_rendered_images = []
        for i in tqdm(range(pred_vertices.shape[0]), desc="Rendering expressions"):
            pred_frame = self.render_exprs_to_image(pred_vertices[i])
            cv2.putText(pred_frame, f"in_motion", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)
            if i in key_frame or (i-pred_vertices.shape[0]) in key_frame:
                id = "%03d"%i
                cv2.putText(pred_frame, f"f:{id} {kf_string}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)
            out_pred_rendered_images.append(pred_frame)


        # create the video out file
        out_vid_file = os.path.join(out_dir, out_seq_name + "_eval_nf%s.mp4" % len(out_pred_rendered_images))
        video_file = self.compose_write_video(out_vid_file, out_pred_rendered_images, self.image_size)

        return video_file, out_pred_rendered_images

    def write_images(self, pred_images_folder, seq_name, pred_frames):
        image_out_folder = os.path.join(pred_images_folder, seq_name)
        os.makedirs(image_out_folder, exist_ok=True)
        print('Writing Images to', image_out_folder)
        for i, frame in tqdm(enumerate(pred_frames), desc="writing pred images"):
            outfile = os.path.join(image_out_folder, "frame%04d.jpg" % i)
            cv2.imwrite(outfile, frame)

    def render_exprs_to_image(self, exprs, shape_params=None):
        """
        exprs:  N x 103 (jawpose, exprs)
        """
        if exprs.shape[0] < 200:
            # print("exprs before pose", exprs.shape)
            jaw_pose = exprs[:3].view(1,-1)
            exprs_only = exprs[3:].view(1,-1)
            # print("Jaw pose", jaw_pose.shape)
            # print("exprs_only pose", exprs_only.shape)
            # print("shape_params", shape_params.shape if shape_params is not None else None)
            flame_vertices = self.face_model.morph(expression_params=exprs_only, jaw_pose=jaw_pose, shape_params=shape_params)[0]
        else:
            flame_vertices = exprs.reshape(-1,3)

        rendered_frame = self.render_images(flame_vertices)
        return rendered_frame

    def render_images(self, vertices):
        # import pdb; pdb.set_trace()
        vertices = vertices.cpu().numpy() + self.cust_trans
        self.face_render.add_face(vertices, self.face_model.faces)
        colour = self.face_render.render()
        return colour

    def compose_write_video(self, out_vid_file, gt_frames,
                            frame_size=(512, 512), fps=30):

        print('Writing video', out_vid_file)
        print()
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        writer = cv2.VideoWriter(out_vid_file, fourcc, fps, frame_size)
        # print("diff frames", len(diff_frames))
        for i, frame in tqdm(enumerate(gt_frames), desc="writing video"):
            out_frame = frame
            writer.write(out_frame)
        writer.release()

        return out_vid_file

    def add_audio_to_video(self, audio_file, video_file, out_vid_w_audio_file):

        ffmpeg_command = f"ffmpeg -y -i {video_file} -i {audio_file} -map 0:v -map 1:a -c:v copy -shortest {out_vid_w_audio_file}"
        print("ffmpeg_command", ffmpeg_command)
        os.system(ffmpeg_command)

        rm_command = ('rm {0} '.format(video_file))
        out = subprocess.call(rm_command.split())
        print("removed ", video_file)

    def dump_mesh(self, vertices, faces, out_file):
        mean_face_mesh = trimesh.Trimesh(vertices, faces, process=False)
        mean_face_mesh.export(out_file)
        

class head_render():

    def __init__(self, render_type="std") -> None:
        
        self.render = render_helper(render_type=render_type)
   
    def render_from_dump(model_path):

        print("Loading the pred files")
        all_files = sorted(glob(os.path.join(model_path, "*.npy")))
        print("all the files to process", len(all_files))

        max_seq_len=576
        pred_poses = {}
        for pred_file in all_files:
            pose = np.load(pred_file, allow_pickle=True)[0] # T x 3
            seq_name = pred_file.split("/")[-1].split(".")[0]
            pred_poses[seq_name] = pose[:max_seq_len]
            break

    def render_single_seq(self, pred_file, audio_file):

        ## load the gt model
        ## apply the head rotation the gt pose
        ## render the video with the normal render
        ## add the audio
        ## render the video with the paper view render
        ## use that for the comparison

        max_seq_len=576
        gt_motion = None
        render_only_exprs = True
        if "sadtalker" in pred_file or "SadTalker" in pred_file:
            print("Running SadTalker", pred_file)
            g_fn =lambda x: scipy.ndimage.filters.gaussian_filter1d(x, sigma=1, axis=0)
            with open(pred_file, 'rb') as fin:
                sub_dict = pickle.load(fin,encoding='latin1')
                pred_pose = g_fn(sub_dict["global_pose"][:max_seq_len])
                gt_motion = sub_dict["vertice"].reshape(-1, 15069)[:max_seq_len]

            seq_name = pred_file.split("/")[-1].split(".")[0]
            seq_name_without_ss = seq_name.split("_ns")[0]
            sample_id = seq_name.split("_ns")[-1][0]
            new_seq_name = seq_name_without_ss + "_ss%02d" % int(sample_id)
            seq_name = new_seq_name

        elif "talkshow" in pred_file or "TalkShow" in pred_file or "TalkSHOW" in pred_file:
            print("Running Talkshow", pred_file)
            seq_name = pred_file.split("/")[-1].split(".")[0]
            seq_name_without_ss = seq_name.split("_ss")[0]
            ss_id = int(seq_name.split("_ss")[1][:2])
            result_dict = np.load(pred_file.replace(seq_name, seq_name_without_ss), allow_pickle=True).item()
            pose = result_dict["head"]
            exprs = result_dict["full"][ss_id][:max_seq_len, 165:265] # 100 params
            jaw = result_dict["full"][ss_id][:max_seq_len, 0:3] # 3 params
            exprs = exprs * 1.06
            gt_motion = self.render.expression_face_model.morph(exprs,jaw_pose=jaw)

            pred_pose = pose[ss_id][:max_seq_len]
            
            # first frame nonormalization
            # pose_matrix = axis_angle_to_matrix(pred_pose)
            # norm_pose_pose_matrix = torch.matmul(torch.linalg.inv(pose_matrix[0]), pose_matrix[:])
            # pred_pose = matrix_to_axis_angle(norm_pose_pose_matrix)
            pred_pose = pred_pose.numpy()
        
        elif "projects/dataset/HDTF/metrical_tracker_dict" in pred_file:
            #render gt
            seq_name = pred_file.split("/")[-1].split(".")[0]
            seq_name_without_ss = seq_name
            pred_pose = "gt"
        
        elif "metric_eval" in pred_file: # done working
            print("Running our model", pred_file)
            pose = np.load(pred_file, allow_pickle=True)[0] # T x 3
            pred_pose = pose[:max_seq_len]
            seq_name = pred_file.split("/")[-1].split(".")[0]
            seq_name_without_ss = seq_name.split("_ss")[0]

        elif "ip_evolution_exp" in pred_file: # visulizing the editing file 
            # print()
            result_dict = np.load(pred_file, allow_pickle=True).item() # T x 3
            last_step_results = result_dict[499]
            pred_pose = last_step_results["sample_with_gt_replc"][:max_seq_len] # T x 3
            # pred_pose[90:250] = 1.8 * pred_pose[90:250]
            seq_name = pred_file.split("/")[-1].replace(".npy", "")
            seq_name_without_ss = seq_name[:-7]
            pred_pose = (pred_pose * std.numpy()) + mean.numpy() 

            render_only_exprs = False

        elif "complete_model_test" in pred_file or "editing_test" in pred_file: # done working
            
            print("Running our model", pred_file)
            result_dict = np.load(pred_file, allow_pickle=True).item() # T x 3
            
            pred_pose = result_dict["pose"][:max_seq_len] # T x 3
            # pred_pose[90:250] = 1.5 * pred_pose[90:250]
            gt_motion = torch.from_numpy(result_dict["vertice"][:max_seq_len].reshape(-1, 15069))

            seq_name = pred_file.split("/")[-1].split(".")[0]
            seq_name_without_ss = seq_name.split("_ss")[0].replace("gt_", "")

            if "swap_kf" in seq_name_without_ss:
                seq_name_without_ss = seq_name_without_ss.split("_swap_kf")[0]

            elif "editing_test" in pred_file and "gt" not in seq_name:
                seq_name_without_ss = seq_name_without_ss[:-7]
            
            if "editing_test" in pred_file:
                render_only_exprs = False

            # first frame nonormalization
            # pose_matrix = axis_angle_to_matrix(torch.from_numpy(pred_pose))
            # norm_pose_pose_matrix = torch.matmul(torch.linalg.inv(pose_matrix[0]), pose_matrix[:])
            # pred_pose = matrix_to_axis_angle(norm_pose_pose_matrix)
            # pred_pose = pred_pose.numpy()
        
            # ### mean frame nonormalization
            # seq_name = seq_name + "_mean_norm"
            # pose_matrix = axis_angle_to_matrix(torch.from_numpy(pred_pose))
            # mean_euler = matrix_to_euler_angles(pose_matrix, "XYZ")
            # mean_euler = torch.mean(mean_euler, dim=0)
            # mean_pose_matrix = euler_angles_to_matrix(mean_euler,"XYZ")
            # norm_pose_pose_matrix = torch.matmul(torch.linalg.inv(mean_pose_matrix), pose_matrix[:])
            # pred_pose = matrix_to_axis_angle(norm_pose_pose_matrix)
            # pred_pose = pred_pose.numpy()
        
        elif "paper_inbetween_" in  pred_file: # pred exprs
            
            gt_motion = np.load(os.path.join(pred_file, "pred.npy"), allow_pickle=True) # T x 3
            # gt_motion = np.load(pred_file, allow_pickle=True) # T x 3
            print("gt_motion", gt_motion.shape)
            gt_motion = gt_motion.reshape(-1, 15069)
            gt_motion[20:130] = scipy.ndimage.filters.gaussian_filter1d(gt_motion[20:130], sigma=0.65, axis=0)
            
            gt_motion = torch.from_numpy(gt_motion[:max_seq_len]).reshape(-1, 15069)
            pred_pose = torch.zeros((max_seq_len, 3)).float().numpy()
            pred_pose = (pred_pose * std.numpy()) + mean.numpy() 
            seq_name = pred_file.split("/")[-1].replace(".npy", "")
            seq_name_without_ss = seq_name.split("_ns")[0]
            render_only_exprs = False

        else: 
            print("pred_file", pred_file)
            raise("Enter a valid experiment type")
        
        if type(pred_pose) is not str:
            print("Load pred", pred_pose.shape)

        if gt_motion is None:
            metrical_tracker_dict = os.path.join(os.environ["DATA_HOME"], "projects/dataset/HDTF", "metrical_tracker_dict")
            gt_file = os.path.join(metrical_tracker_dict, seq_name_without_ss+".pkl")
            with open(gt_file, 'rb') as fin:
                sub_dict = pickle.load(fin,encoding='latin1')
                gt_motion = sub_dict["vertice"].reshape(-1, 15069)[:max_seq_len]

                if pred_pose == "gt":
                    pred_file = gt_file
                    g_fn =lambda x: scipy.ndimage.filters.gaussian_filter1d(x, sigma=2, axis=0)
                    pred_pose = sub_dict["global_pose"][:max_seq_len].numpy()
                    pred_pose = g_fn(pred_pose)

        print("Load gt", gt_motion.shape)

        if audio_file is None:
            if "hdtf" in pred_file or "HDTF" in pred_file or "ip_evolution_exp" in pred_file:
                audio_path = os.path.join(os.getenv("DATA_HOME"), "projects/dataset/HDTF", "wav")
                audio_file = os.path.join(audio_path, seq_name_without_ss+".wav")     
            elif "external_audio_test" in  pred_file or "external_test_audio" in pred_file or "editing_test" in pred_file:
                print("audio_path", os.getenv("DATA_HOME"))
                audio_path = os.path.join(os.getenv("DATA_HOME"), "projects/dataset/external_audio_test/head_motion_test", "wav")
                audio_file = os.path.join(audio_path, seq_name_without_ss+".wav")     
            elif "paper_inbetween_" in pred_file:
                audio_path = os.path.join(os.getenv("DATA_HOME"), "projects/dataset/tracker_dataset", "wav")
                audio_file = os.path.join(audio_path, seq_name_without_ss+".wav")

        out_dir = os.path.dirname(pred_file)+"_video"
        if "blender" in self.render.render_type:
            out_dir = out_dir.replace("_video", f"_{self.render.render_type}_video")
        os.makedirs(out_dir, exist_ok=True)

        pred_pose = torch.from_numpy(pred_pose)
        seq_len =  min(gt_motion.shape[0], pred_pose.shape[0])

        gt_motion = gt_motion[:seq_len]
        pred_pose = pred_pose[:seq_len]

        if  "talkshow" in pred_file or "TalkShow" in pred_file or "TalkSHOW" in pred_file:
             curr_pred = self.render.expression_face_model.apply_neck_rotation(gt_motion, pred_pose).detach()
        else:
            curr_pred = self.render.face_model.apply_neck_rotation(gt_motion, pred_pose).detach()


        nf = 600
        curr_pred = curr_pred[:nf]
        out_file, pred_rendered_images = self.render.visualize_meshes(out_dir, seq_name,
                                                                       audio_file, curr_pred,
                                                                       None)
        if render_only_exprs:
            out_file, pred_rendered_images = self.render.visualize_meshes(out_dir, seq_name+"_only_exprs",
                                                            audio_file, gt_motion[:nf],
                                                            None)
        # out_file, pred_rendered_images = self.render.visualize_meshes(out_dir, seq_name+"_only_exprs",
        #                                         audio_file, gt_motion[:nf],
        #                                         None)

def add_local_arguments(parser):
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--render_type", type=str, default="blender_800")
    parser.add_argument("--seq", type=str, default=None)
    parser.add_argument("--audio_file", type=str, default=None)

    parser.set_defaults(unseen=False)
    return parser

if __name__ == "__main__":
    parser = ArgumentParser()
    add_local_arguments(parser)
    args=parse_and_load_from_model(parser)

    if args.model_path is None:
        Warning("using debug mode")
        # args.model_path = "/home/bthambiraja/fast/projects/motion_root/logs/head_reg_16_ip_mp0_02_rd02_3D_stdnorm_waud_skip_vel_loss0010/complete_model_test_user138_external_test_audio_model000100035_gen_model/npy"
        # args.seq = "01welcome_ss00.npy"
        # args.model_path = "/is/rg/ncs/projects/bthambiraja/SadTalker_hdtf/processed_to_voca_format"
        # args.model_path = "/is/rg/ncs/projects/bthambiraja/TalkShow/hdtf_subj_wise"
        # os.environ["DATA_HOME"] = os.path.join(os.environ["DATA_HOME"], "work")

        # ## visulize the edit
        # args.model_path = "/is/rg/ncs/projects/bthambiraja/submission_models/head_reg_13_nomask_mp0_5_rd02_3D_stdnorm_waud/2024-03-12T20-23-16_conditional_gen_no_stoch_comp_unseen_uncond_model000100035_ip_evolution_exp/npy_dump"
        # args.seq = "RD_Radio11_000030_090.npy"
        # # args.seq = "RD_Radio10_000030_090.npy"

        # ## render exprs
        # args.model_path = "/home/bthambiraja/fast/projects/motion_root/submission_models/cond_012_17_00_concat_prob10_vel10/2023-11-24T15-09-50_paper_inbetween_uncond_exp_guid_0.99_model000050000_cond03/npy"
        # args.seq = "boris_corona_only_exprs_naive_cut_sentence03_ns_00.npy"

        ##
        # args.model_path = "/is/rg/ncs/projects/bthambiraja/submission_models/head_reg_16_ip_mp0_02_rd02_3D_stdnorm_waud_skip_vel_loss0010/complete_model_test_HDTF_user_138_edit_model000100035_editing_test/npy"
        # args.seq = "RD_Radio34_003090_250_ss02.npy"

        args.model_path = "/is/rg/ncs/projects/bthambiraja/submission_models/head_reg_16_ip_mp0_02_rd02_3D_stdnorm_waud_skip_vel_loss0010/complete_model_test_user_138_external_test_audio_model_per_model_a4K_gf99_gen_gf0.3_model000100035/npy"
        args.seq = "wD-jLNmRVfw_ss00.npy"


        # ## render_new_mesh
        # args.model_path = "/home/bthambiraja/fast/projects/motion_root/logs_bkp_feb_05_24/sa_08_04_fo_trevor_noah_win900_vel01_win30/2023-11-24T16-04-12_paper_inbetween_uncond_exp_guid_0.99_model000009000_cond02"
        # args.seq = "trevor_noah_exp_naive_cut_sentence03_ns_00"


        # args.model_path = "/is/rg/ncs/projects/bthambiraja/SadTalker/external_audio_test/processed_to_voca_format"
        # args.model_path = "/is/rg/ncs/projects/bthambiraja/TalkShow/hdtf_subj_wise"
        # args.model_path = "/is/rg/ncs/projects/bthambiraja/TalkShow/external_audio_test"
        os.environ["DATA_HOME"] = os.path.join(os.environ["DATA_HOME"], "work")
        # args.seq = "wD-jLNmRVfw.pkl"
        args.render_type = "ratio_800x1200"
        os.environ["BLENDER_PATH"] = "/snap/bin/blender"
    
    if args.seq is None:
        raise("Invalid seq")
        
    tester = head_render(args.render_type)
    tester.render_single_seq(os.path.join(args.model_path, args.seq), args.audio_file)
    # args.seq = "boris_corona_only_exprs_naive_cut_sentence01_ns_00.npy"
    # tester.render_single_seq(os.path.join(args.model_path, args.seq), args.audio_file)