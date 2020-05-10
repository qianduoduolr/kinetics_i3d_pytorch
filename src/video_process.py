import torch
import numpy as np

import random
import os
import os.path
import argparse
import cv2
import pickle

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))



def load_rgb_frames(image_dir, vid, clip, dataset='data1'):

  frames = []
  for i in clip:
    try:
        if dataset == 'data1' or dataset == 'data3':
            img = cv2.imread(os.path.join(image_dir, vid, 'frame{}.jpg'.format(i)))[:, :, [2, 1, 0]]
        else:
            num_len = len(str(i))
            num_frame = vid + '-' + '0' * (6-num_len) + str(i) + '.jpg'
            frame_file = os.path.join(image_dir,vid,num_frame)
            img = cv2.imread(frame_file)[:, :, [2,1,0]]
        w,h,c = img.shape
        if w < 226 or h < 226:
            d = 226.-min(w,h)
            sc = 1+d/min(w,h)
            img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
        if dataset == 'data3':
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = (img/255.)*2 - 1
        frames.append(img)
    except Exception:
        continue
  return np.asarray(frames, dtype=np.float32)


def save_to_numpy(args):


    video_dir = args.raw_data_path
    save_dir = args.process_data_path
    rename = args.rename
    dic = {}
    video_files = []
    count = 1
    for classes in os.listdir(video_dir):
        if classes != '.DS_Store':
            for vid_file in os.listdir(os.path.join(video_dir,classes)):

                if vid_file[-3:] == 'flv':
                    if rename == 'yes':
                        try:
                            src = os.path.join(os.path.join(video_dir,classes),vid_file)
                            dst = os.path.join(os.path.join(video_dir,classes),'{}.flv'.format(count))
                            os.rename(src,dst)
                            ass_file = vid_file[:-3] + 'ass'
                            src = os.path.join(os.path.join(video_dir,classes),ass_file)
                            dst = os.path.join(os.path.join(video_dir,classes),'{}.ass'.format(count))
                            os.rename(src,dst)
                            xml_file = vid_file[:-3] + 'xml'
                            src = os.path.join(os.path.join(video_dir,classes),xml_file)
                            dst = os.path.join(os.path.join(video_dir,classes),'{}.xml'.format(count))
                            os.rename(src,dst)
                            dic[vid_file[:-4]] = count
                            count += 1
                        except Exception as e:
                            print(e)
                    else:
                        video_files.append(vid_file)
            #
            # for idx,vid in enumerate(video_files):
            #
            #     file = os.path.join(video_dir,vid)
            #     video_to_numpy(file,save_dir,idx,vid)
            #     # name = "{}_{}_{}".format(idx+1, fps, duration)
            #     print('finish {}/{} videos frame spliting!'.format(idx,len(video_files)))

                # np.save(os.path.join(save_dir,name),data)
    with open('/Users/lr/Desktop/VMR/data/dataset/name_to_num.pickle','wb') as w:
        pickle.dump(dic,w)

def video_to_numpy(file,save_dir,idx,vid):
    cap = cv2.VideoCapture(file)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    time = count / fps

    # save_file = os.path.join(save_dir,'{}_{}_{}'.format(idx+1,int(fps),int(time)))
    save_file = os.path.join(save_dir,vid)
    if not os.path.exists(save_file):
        os.mkdir(save_file)

    count = 0
    success = True
    while success:
        success, image = cap.read()
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(save_file,"frame{}.jpg".format(count)), image)  # save frame as JPEG file

        count += 1

def make_dataset(processed_file,num,timestamp = 10):

    data = []
    start_video = num[0]
    end_video = num[1]
    for vid in os.listdir(processed_file):
        if vid != '.DS_Store':
            idx, fps, time = list(vid.split('_'))
            idx = int(idx)
            if idx >= end_video or idx < start_video:
                continue
            fps = int(float(fps))
            time = int(float(time))
            file = os.path.join(processed_file,vid)
            num_frame = len(os.listdir(file))
            if num_frame <= 16:
                continue
            interval = fps * timestamp
            if num_frame <= interval * 3:
                candidate = list(range(num_frame))
                x = num_frame // interval
                if x == 0:
                    data.append((candidate,vid))

                else:
                    for t in range(x):
                        clip = candidate[t*interval:(t+1)*interval]
                        data.append((clip,vid))
            else:

                candidate = list(range(num_frame))
                t = candidate[::interval]
                for start in t[1:-1]:

                    clip = candidate[start:start+interval]
                    data.append((clip,vid))

    return  data


def make_dataset_sta(processed_file, record_file,num):
    data = []
    start_video = num[0]
    end_video = num[1]
    with open(record_file,'r') as f:
        lines = f.readlines()
        for line in lines[start_video:end_video]:
            l = line.split('##')
            vid, start, end = l[0].split()
            start_frame = int(24 * float(start)) + 1
            end_frame = int(24 * float(end)) + 1
            if end_frame - start_frame < 16:
                continue
            clip = list(range(start_frame,end_frame))
            data.append((clip,vid))

    return data

def make_dataset_tos(processed_file,num,timestamp = 8):

    data = []
    start_video = num[0]
    end_video = num[1]
    vids = sorted(os.listdir(processed_file))
    vids = vids[start_video:end_video]
    for vid in vids:
        if vid != '.DS_Store':
            # if idx >= end_video or idx < start_video:
            #     continue
            # fps = int(float(fps))
            # time = int(float(time))
            file = os.path.join(processed_file,vid)
            num_frame = len(os.listdir(file))
            if num_frame <= 16:
                continue
            interval = 30 * timestamp
            if num_frame <= interval * 3:
                candidate = list(range(num_frame))
                x = num_frame // interval
                if x == 0:
                    data.append((candidate,vid))

                else:
                    for t in range(x):
                        clip = candidate[t*interval:(t+1)*interval]
                        data.append((clip,vid))
            else:

                candidate = list(range(num_frame))
                t = candidate[::interval]
                for start in t[1:-1]:

                    clip = candidate[start:start+interval]
                    data.append((clip,vid))

    return  data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--raw-data-path',
        type=str,
        default='/Users/lr/Desktop/VMR/data/dataset/raw_dataset/videos-1',
        help='Path to raw videos')
    parser.add_argument(
        '--process-data-path',
        type=str,
        default='/Users/lr/Desktop/VMR/data/TACoS/video',
        help='Path to processed videos')

    parser.add_argument(
        '--rename',
        type=str,
        default='yes',
        help='whether rename')
    args = parser.parse_args()
    save_to_numpy(args)
