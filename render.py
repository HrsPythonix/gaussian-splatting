#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import concurrent.futures
import time
from PIL import Image
import json

def save_image(image, path):
    # torchvision.utils.save_image(image, path)
    ndarr = image.numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, skip_loading):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        total_start = time.time()
        render_time_count = 0
        submit_time_count = 0
        pp_time_count = 0
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            render_start = time.time()
            rendering = render(view, gaussians, pipeline, background)["render"]
            render_end = time.time()
            render_time_count += render_end - render_start

            pp_start = time.time()
            image = rendering.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8)
            #gt = view.original_image[0:3, :, :].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8)
            pp_end = time.time()
            pp_time_count += pp_end - pp_start

            sub_start = time.time()
            executor.submit(save_image, image, os.path.join(render_path, view.image_name + '.png'))
            #executor.submit(save_image, gt, os.path.join(gts_path, view.image_name + '.png'))
            sub_end = time.time()
            submit_time_count += sub_end - sub_start

            # save_start = time.time()
            # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            # if not skip_loading:
            #     gt = view.original_image[0:3, :, :]
            #     torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            # save_end = time.time()
            # print("save:", save_end - save_start)
        print("render_time_count", render_time_count/len(views))
        print("pp_time_count", pp_time_count/len(views))
        print("submit_time_count", submit_time_count/len(views))
        print("All jobs submitted, waiting IO......")

        io_wait_start = time.time()
        executor.shutdown(wait=True)
        total_end = time.time()
        print(f"IO Finished! IO wait {total_end - io_wait_start}s, All time cost: {total_end - total_start}s")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_loading : bool, save_pose : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_loading=skip_loading)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, skip_loading)
            
            if save_pose:
                pose_info = []
                for idx, view in enumerate(scene.getTrainCameras()):
                    info = {}
                    info['image_name'] = view.image_name + '.png'
                    info['colmap_id'] = view.colmap_id
                    info['R'] = view.R.tolist()
                    info['T'] = view.T.tolist()
                    info['FoVx'] = view.FoVx
                    info['FoVy'] = view.FoVy
                    info['zfar'] = view.zfar
                    info['znear'] = view.znear

                    info['world_view_transform'] = view.world_view_transform.to("cpu").numpy().tolist()
                    info['projection_matrix'] = view.projection_matrix.to("cpu").numpy().tolist()

                    pose_info.append(info)
                with open(os.path.join(dataset.model_path, "train", "ours_{}".format(scene.loaded_iter), "train.json"), 'w') as json_f:
                    json.dump(pose_info, json_f, indent=4)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, skip_loading)
            
            if save_pose:
                pose_info = []
                for idx, view in enumerate(scene.getTestCameras()):
                    info = {}
                    info['image_name'] = view.image_name + '.png'
                    info['colmap_id'] = view.colmap_id
                    info['R'] = view.R.tolist()
                    info['T'] = view.T.tolist()
                    info['FoVx'] = view.FoVx
                    info['FoVy'] = view.FoVy
                    info['zfar'] = view.zfar
                    info['znear'] = view.znear

                    info['world_view_transform'] = view.world_view_transform.to("cpu").numpy().tolist()
                    info['projection_matrix'] = view.projection_matrix.to("cpu").numpy().tolist()

                    pose_info.append(info)
                with open(os.path.join(dataset.model_path, "test", "ours_{}".format(scene.loaded_iter), "test.json"), 'w') as json_f:
                    json.dump(pose_info, json_f, indent=4)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_loading", action="store_true", default=False)
    parser.add_argument("--save_pose", action="store_true", default=False)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    #torch.cuda.set_per_process_memory_fraction(0.3, 0)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_loading, args.save_pose)