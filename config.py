import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--continue_training", action="store_true")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--scheduler_milestones", type=list, default=[100, 200, 300, 400])
    parser.add_argument("--scheduler_lambda", type=float, default=0.1)
    parser.add_argument("--attack_choice", type=str, default="any2any")
    parser.add_argument("--n_iters", type=int, default=350)
    parser.add_argument("--num_workers", type=float, default=4)
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--attack_ratio", type=int, default=60)
    parser.add_argument("--transform_ratio", type=int, default=60)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--s", type=int, default=1)
    parser.add_argument("--lr_iter", type=int, default=30)
    parser.add_argument("--aug", type=str, default="flowaug")
    return parser