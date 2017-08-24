import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src import inflate
from src.i3d import I3DenseNet


@profile
def run_inflater():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageFolder('/sequoia/data1/yhasson/datasets/test-dataset',
            transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    frame_nb = 4
    densenet = torchvision.models.densenet121(pretrained=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    i3densenet = I3DenseNet(densenet, frame_nb, inflate_block_convs=False)
    for i, (input_2d, target) in enumerate(loader):
        target = target.cuda()
        target_var = torch.autograd.Variable(target)
        input_2d_var = torch.autograd.Variable(input_2d)
        out2d = densenet(input_2d_var)
        
        input_3d = input_2d.unsqueeze(2).repeat(1, 1, frame_nb, 1, 1)
        input_3d_var = torch.autograd.Variable(input_3d) 
        out3d = i3densenet(input_3d_var)

        out_diff = out2d.data - out3d.data
        print(out_diff.max())
        assert(out_diff.max() < 0.0001)


if __name__ == "__main__":
    run_inflater()