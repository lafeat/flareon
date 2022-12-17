import json
import shutil
import config
import numpy as np
import copy
import torch
import torch.nn.functional as F
import torchvision
import kornia.augmentation as A
from model.resnet32 import ResNet18
from model.resnet64 import Resnet18
from augment.Cutout import Cutout
from network.models import Denormalizer
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader import PostTensorTransform, get_dataloader
from utils.utils import progress_bar
import os


def train(net, optimizer, optimizer_ins, scheduler, train_dl, identity_grid, ins, tf_writer, epoch, opt):
    print(" Train:")
    # device_ids = [0, 1, 2, 3]
    net = torch.nn.DataParallel(net)
    net.train()
    total_loss_ce = 0
    total_sample = 0
    total_clean_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()
    denormalizer = Denormalizer(opt)
    transforms = PostTensorTransform(opt).to(opt.device)
    rand_crop = A.RandomCrop((opt.input_height, opt.input_width), padding=4)
    cut = Cutout(n_holes=1, length=16)
    warmup = opt.warmup_epochs

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizer.zero_grad()
        optimizer_ins.zero_grad()
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]
        num_bd = bs * opt.attack_ratio // 100
        if opt.dataset == "cifar10":
            inputs = rand_crop(inputs)

        if opt.aug == "flowaug":
            inputs[bs * opt.transform_ratio // 100:] = transforms(inputs[bs * opt.transform_ratio // 100:])

        if opt.attack_choice == "any2any":
            input_origin = copy.deepcopy(inputs)
            igrid = identity_grid.repeat(num_bd, 1, 1, 1)
            attack_inputs, clean_inputs = inputs.split([num_bd, bs - num_bd])
            attack_labels, clean_labels = targets.split([num_bd, bs - num_bd])
            ains = ins[attack_labels]
            ains = torch.squeeze(ains, dim=1)
            grid_temp = igrid + ains / opt.input_height
            grid_temp = torch.clamp(grid_temp, -1, 1).float()

            attacked_inputs = F.grid_sample(attack_inputs, grid_temp, align_corners=True)
            inputs = torch.cat((attacked_inputs, clean_inputs), dim=0)
            targets = torch.cat((attack_labels, clean_labels), dim=0)

        elif opt.attack_choice == "any2one":
            target_label_index = targets == opt.target_label
            attack_inputs, clean_inputs = inputs[target_label_index], inputs[~target_label_index]
            attack_targets, clean_targets = targets[target_label_index], targets[~target_label_index]
            targets = torch.cat((attack_targets, clean_targets), dim=0)
            input_origin = copy.deepcopy(torch.cat((attack_inputs, clean_inputs), dim=0))
            num_any2one = attack_inputs.shape[0]
            grid_temp = identity_grid + ins / opt.input_height
            grid_temp = torch.clamp(grid_temp, -1, 1).float()

            attacked_inputs = F.grid_sample(attack_inputs, grid_temp.repeat(num_any2one, 1, 1, 1), align_corners=True)
            inputs = torch.cat((attacked_inputs, clean_inputs), dim=0)

        if opt.dataset == "cifar10":
            inputs = cut(inputs)

        preds = net(inputs)
        loss_ce = criterion_CE(preds, targets)
        loss = loss_ce
        # loss = loss_ce + mse_reg * F.l1_loss(poisoned_inputs1, poisoned_inputs2)
        # loss += l1_reg * ains.abs().mean()
        loss.backward()
        optimizer.step()
        constrain = ins.norm(2, (1, 2, 3)).mean()

        if epoch < warmup and constrain > opt.eps:
            optimizer_ins.step()
        else:
            pass

        total_loss_ce += loss_ce.detach()
        total_sample += bs
        total_clean_correct += torch.sum(torch.argmax(preds, dim=1) == targets)
        avg_acc_clean = total_clean_correct * 100.0 / total_sample
        avg_loss_ce = total_loss_ce / total_sample
        progress_bar(batch_idx, len(train_dl),
                     "CE Loss: {:.4f} | Clean Acc: {:.4f}".format(avg_loss_ce, avg_acc_clean))

        if batch_idx == len(train_dl) - 2:
            if num_bd > 0:
                residual = inputs[:num_bd] - input_origin[:num_bd]
                batch_img = torch.cat([input_origin[:num_bd], inputs[:num_bd], residual], dim=2)
                batch_img = denormalizer(batch_img)
                batch_img = F.interpolate(batch_img, scale_factor=(4, 4))
                grid = torchvision.utils.make_grid(batch_img, normalize=True)
            else:
                batch_img = inputs
                batch_img = denormalizer(batch_img)
                grid = torchvision.utils.make_grid(batch_img, normalize=True)


    if not epoch % 1:
        tf_writer.add_scalars("Clean Accuracy", {"Clean": avg_acc_clean}, epoch)
        tf_writer.add_image("Images", grid, global_step=epoch)
    scheduler.step()


def eval(net, optimizer, scheduler, test_dl, identity_grid, ins, best_clean_acc, best_bd_acc, tf_writer, epoch, opt,
         count):
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

            # tg_label = random.randint(0, opt.num_classes - 1)
            inputs_bd = inputs
            # ains = ins * opt.norms / ins.norm(1, (2, 3, 4), keepdim=True).repeat(1, 1, opt.input_height, opt.input_height, 1)
            if opt.attack_choice == "any2any":
                grid_temps = (identity_grid + ins[opt.target_label] / opt.input_height)
            else:
                grid_temps = (identity_grid + ins / opt.input_height)
            grid_temps = torch.clamp(grid_temps, -1, 1).float()
            inputs_bd = F.grid_sample(inputs_bd, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
            preds_bd = net(inputs_bd)
            targets_bd = torch.ones_like(targets) * opt.target_label

            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample
            info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(acc_clean,
                                                                                                    best_clean_acc,
                                                                                                    acc_bd,
                                                                                                    best_bd_acc)
            progress_bar(batch_idx, len(test_dl), info_string)

    # tensorboard
    if not epoch % 1:
        tf_writer.add_scalars("Test Accuracy", {"Clean": acc_clean, "Bd": acc_bd}, epoch)

    # Save checkpoint
    if (acc_clean > best_clean_acc and acc_bd > best_bd_acc - 0.3):
        count = 0
        print(" Saving...")
        best_clean_acc = acc_clean
        best_bd_acc = acc_bd
        state_dict = {
            "net": net.state_dict(),
            "scheduler": scheduler.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_clean_acc": best_clean_acc,
            "best_bd_acc": best_bd_acc,
            "epoch_current": epoch,
            "identity_grid": identity_grid,
            "ins": ins,
        }
        torch.save(state_dict, opt.ckpt_path)
        with open(os.path.join(opt.ckpt_folder, "Results.txt"), "w+") as f:
            results_dict = {
                "clean_acc": best_clean_acc.item(),
                "bd_acc": best_bd_acc.item()
            }
            json.dump(results_dict, f, indent=2)

    return best_clean_acc, best_bd_acc, count


def main():
    opt = config.get_arguments().parse_args()
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

    # Dataset
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)

    if opt.attack_choice == "any2any":
        ins = np.random.beta(opt.s, opt.s, (opt.num_classes, 1, opt.input_height, opt.input_height, 2)) * 2 - 1
        ins = torch.tensor(ins).to(opt.device)
    elif opt.attack_choice == "any2one":
        ins = np.random.beta(opt.s, opt.s, (1, opt.input_height, opt.input_height, 2)) * 2 - 1
        ins = torch.tensor(ins).to(opt.device)
    ins = torch.nn.Parameter(ins.clone().detach().requires_grad_(True)).to(opt.device)

    # prepare model
    if opt.dataset == "cifar10":
        net = ResNet18(num_classes=opt.num_classes).to(opt.device)
    if opt.dataset == "celeba":
        net = Resnet18(num_classes=opt.num_classes).to(opt.device)
    if opt.dataset == "tinyimagenet":
        net = Resnet18(num_classes=opt.num_classes).to(opt.device)

    if opt.attack_choice == "any2any":
        optimizer_ins = torch.optim.SGD([ins], lr=0.2, momentum=0.9, weight_decay=5e-4)
        optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=5e-4)
    elif opt.attack_choice == "any2one":
        optimizer_ins = torch.optim.SGD([ins], lr=0.2, momentum=0.9, weight_decay=5e-4)
        optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.scheduler_milestones, opt.scheduler_lambda)

    # Load pretrained model
    opt.ckpt_folder = os.path.join(opt.checkpoints,
                                   'ResNet18_warm={}_train_learn_{}_at_ratio={}_aug_ratio={}_s={}_attack_choice={}'.format
                                   (opt.warmup_epochs, opt.dataset, opt.attack_ratio,
                                    opt.transform_ratio, opt.s, opt.attack_choice))
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "train_learn_{}.pth.tar".format(opt.dataset))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    if opt.continue_training:
        if os.path.exists(opt.ckpt_path):
            print("Continue training")
            state_dict = torch.load(opt.ckpt_path)
            net.load_state_dict(state_dict["net"])
            optimizer.load_state_dict(state_dict["optimizer"])
            scheduler.load_state_dict(state_dict["scheduler"])
            best_clean_acc = state_dict["best_clean_acc"]
            best_bd_acc = state_dict["best_bd_acc"]
            epoch_current = state_dict["epoch_current"]
            identity_grid = state_dict["identity_grid"]
            ins = state_dict["ins"]
            tf_writer = SummaryWriter(log_dir=opt.log_dir)
            ins = torch.nn.Parameter(ins.clone().detach().requires_grad_(True)).to(opt.device)
            optimizer_ins = torch.optim.SGD([ins], lr=0.05, momentum=0.9, weight_decay=5e-4)
        else:
            print("Pretrained model doesnt exist")
            exit()
    else:
        print("Train from scratch")
        best_clean_acc = 0.0
        best_bd_acc = 0.0
        epoch_current = 0

        # Prepare grid
        array1d = torch.linspace(-1, 1, steps=opt.input_height)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), dim=2)[None, ...].to(opt.device)

        shutil.rmtree(opt.ckpt_folder, ignore_errors=True)
        os.makedirs(opt.log_dir)
        with open(os.path.join(opt.ckpt_folder, "opt.json"), "w+") as f:
            json.dump(opt.__dict__, f, indent=2)
        tf_writer = SummaryWriter(log_dir=opt.log_dir)

    count = 0
    for epoch in range(epoch_current, opt.n_iters):
        print("Epoch {}:".format(epoch + 1))
        train(net, optimizer, optimizer_ins, scheduler, train_dl, identity_grid, ins, tf_writer, epoch, opt)
        best_clean_acc, best_bd_acc, count = eval(net, optimizer, scheduler, test_dl, identity_grid, ins,
                                                  best_clean_acc, best_bd_acc, tf_writer, epoch, opt, count)
        count = count + 1
        print(count)
        print(opt.lr)
        if count == opt.lr_iter:
            opt.lr = opt.lr / 2
            count = 0


if __name__ == "__main__":
    main()