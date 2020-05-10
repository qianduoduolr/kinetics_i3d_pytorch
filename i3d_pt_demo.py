import argparse

import numpy as np
import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

from src.i3dpt import I3D
from src.test_dataset import *
from src.videotransforms import *
from torchvision import datasets, transforms
import pickle

rgb_pt_checkpoint = 'model/model_rgb.pth'


def run_demo(args):
    print('getting start !')
    args.rgb = 1
    kinetics_classes = [x.strip() for x in open(args.classes_path)]
    score_list = []

    def get_scores(sample, model):
        # sample_var = torch.autograd.Variable(torch.from_numpy(sample))
        sample_var = sample

        out_var, out_logit = model(sample_var)
        out_tensor = out_var.data.cpu()

        top_val, top_idx = torch.sort(out_tensor, 1, descending=True)

        for i in range(top_idx.shape[0]):

            score_list.append(out_tensor[i].numpy().tolist())

        return out_logit

    # load data
    file = args.process_data_path
    test_transforms = transforms.Compose([CenterCrop(224)])
    dataset = Danmus(file, args.num, args.record_file, transforms= test_transforms, dataset = args.dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                             pin_memory=True,drop_last=True)
    # Rung RGB model
    if args.rgb:
        i3d_rgb = I3D(num_classes=400, modality='rgb')
        i3d_rgb.eval()
        i3d_rgb.load_state_dict(torch.load(args.rgb_weights_path))
        # i3d_rgb.cuda()
        with torch.no_grad():
            for idx, rgb_sample in enumerate(dataloader):
                frame_count = rgb_sample.size()
                print(frame_count)
                if frame_count[2] <= 16:
                    continue
                out_rgb_logit = get_scores(rgb_sample, i3d_rgb)
                print('finish {}/{} video segments classfication!'.format((idx+1)*args.batch_size,len(dataloader)*args.batch_size))

    # Run flow model
    if args.flow:
        i3d_flow = I3D(num_classes=400, modality='flow')
        i3d_flow.eval()
        i3d_flow.load_state_dict(torch.load(args.flow_weights_path))
        i3d_flow.cuda()

        flow_sample = np.load(args.flow_sample_path).transpose(0, 4, 1, 2, 3)
        out_flow_logit = get_scores(flow_sample, i3d_flow)

    # Joint model
    if args.flow and args.rgb:
        out_logit = out_rgb_logit + out_flow_logit
        out_softmax = torch.nn.functional.softmax(out_logit, 1).data.cpu()
        top_val, top_idx = torch.sort(out_softmax, 1, descending=True)

        print('===== Final predictions ====')
        print('logits proba class '.format(args.top_k))
        for i in range(args.top_k):
            logit_score = out_logit[0, top_idx[0, i]].data.item()
            print('{:.6e} {:.6e} {}'.format(logit_score, top_val[0, i],
                                            kinetics_classes[top_idx[0, i]]))

    # Save top list
    file = 'data/score_list_{}_{}_{}.pickle'.format(args.num[0],args.num[1],args.dataset)
    with open(file,'wb') as writer:
        pickle.dump(score_list,writer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Runs inflated inception v1 network on\
    cricket sample from tensorflow demo (generate the network weights with\
    i3d_tf_to_pt.py first)')

    # RGB arguments
    parser.add_argument(
        '--rgb', action='store_true', help='Evaluate RGB pretrained network')
    parser.add_argument(
        '--rgb-weights-path',
        type=str,
        default='model/model_rgb.pth',
        help='Path to rgb model state_dict')
    parser.add_argument(
        '--rgb-sample-path',
        type=str,
        default='data/kinetic-samples/v_CricketShot_g04_c01_rgb.npy',
        help='Path to kinetics rgb numpy sample')

    # Flow arguments
    parser.add_argument(
        '--flow', action='store_true', help='Evaluate flow pretrained network')
    parser.add_argument(
        '--flow-weights-path',
        type=str,
        default='model/model_flow.pth',
        help='Path to flow model state_dict')
    parser.add_argument(
        '--flow-sample-path',
        type=str,
        default='data/kinetic-samples/v_CricketShot_g04_c01_flow.npy',
        help='Path to kinetics flow numpy sample')

    parser.add_argument(
        '--classes-path',
        type=str,
        default='data/kinetic-samples/label_map.txt',
        help='Number of video_frames to use (should be a multiple of 8)')
    parser.add_argument(
        '--top-k',
        type=int,
        default='5',
        help='When display_samples, number of top classes to display')
    parser.add_argument(
        '--process-data-path',
        type=str,
        default='/Users/lr/Desktop/VMR/data/dataset/processed_data',
        help='Path to processed videos')
    parser.add_argument(
        '--record-file',
        type=str,
        default='/data/656146095/Danmu/charades_sta_train.txt',
        help='Path to records')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='batchsize')
    parser.add_argument(
        '--num',
        nargs= '+',
        type=int,
        default=[0,1400],
        help='video nums')
    parser.add_argument(
        '--dataset',
        type=str,
        default='data1',
        help='video dataset')
    args = parser.parse_args()
    run_demo(args)
