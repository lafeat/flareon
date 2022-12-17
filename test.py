import config
import torch
import random
import numpy as np
import torch.nn.functional as F
from model.resnet32 import ResNet18
from model.resnet64 import Resnet18
from utils.dataloader import get_dataloader
from utils.utils import progress_bar
import os

def eval(net, test_dl, identity_grid, ins, opt):
    print(" Eval:")
    net.to(opt.device)
    net.eval()
    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = net(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            inputs_bd = inputs
            if opt.attack_choice == 'any2any':
                grid_temps = (identity_grid + ins[opt.target_label] / opt.input_height)
            elif opt.attack_choice == 'any2one':
                grid_temps = (identity_grid + ins / opt.input_height)
            grid_temps = torch.clamp(grid_temps, -1, 1).float()
            inputs_bd = F.grid_sample(inputs_bd, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
            preds_bd = net(inputs_bd)
            targets_bd = torch.ones_like(targets) * opt.target_label

            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
            clean_acc = total_clean_correct * 100.0 / total_sample
            bd_acc = total_bd_correct * 100.0 / total_sample
            info_string = "Clean Acc: {:.4f}| Bd Acc: {:.4f}".format(
                clean_acc, bd_acc)
            progress_bar(batch_idx, len(test_dl), info_string)


def main():
    opt = config.get_arguments().parse_args()
    if opt.dataset == "cifar10":
        net = ResNet18(num_classes=opt.num_classes).to(opt.device)
    if opt.dataset == "celeba":
        net = Resnet18(num_classes=opt.num_classes).to(opt.device)
    if opt.dataset == "tinyimagenet":
        net = Resnet18(num_classes=opt.num_classes).to(opt.device)

    if opt.dataset == "cifar10":
        opt.num_classes = 10
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "celeba":
        opt.num_classes = 8
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    elif opt.dataset == "tinyimagenet":
        opt.num_classes = 200
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")


    # Dataset and model
    test_dl = get_dataloader(opt, False)


    # pretrained model
    opt.ckpt_folder = os.path.join(opt.checkpoints,
                                   'ResNet18_{}-at_ratio={}-ag_ratio={}-s={}-mode={}'.format(
                                       opt.dataset, opt.attack_ratio, opt.transform_ratio, opt.s, opt.attack_choice))

    # opt.ckpt_folder = os.path.join(opt.checkpoints,
    #                                'ResNet18_warm={}_train_learn_{}_at_ratio={}_ag_ratio={}_s={}_attack_choice={}'.format
    #                                (opt.warmup_epochs, opt.dataset, opt.attack_ratio,
    #                                 opt.transform_ratio, opt.s, opt.attack_choice))

    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}.pth.tar".format(opt.dataset))
    # opt.ckpt_path = os.path.join(opt.ckpt_folder, "train_learn_{}.pth.tar".format(opt.dataset))

    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    state_dict = torch.load(opt.ckpt_path)
    net.load_state_dict(state_dict["net"])
    identity_grid = state_dict["identity_grid"]
    ins = state_dict["ins"]
    eval(net, test_dl, identity_grid, ins, opt)



if __name__ == "__main__":
    main()

