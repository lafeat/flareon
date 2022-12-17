# Flareon: Stealthy Backdoor Injection via Poisoned Augmentation

## Introdution

This is the official release
of "Flareon: Stealthy Backdoor Injection via Poisoned Augmentation."

![High-level overview](https://github.com/lafeat/flareon/blob/main/asset/overview.png)


## Requirements

- Install required python packages:
```bash
$ python -m pip install -r requirements.py
```

## Training
Training commands are as follows.

* Any-to-any:
```bash
$ python train.py --dataset <dataset name> --attack_ratio <ratio> --aug <augment> --s <beta>
```
* Adaptive any-to-any:
```bash
$ python train_learn.py --dataset <dataset name> --attack_ratio <ratio> --aug <augment> --s <beta> --warmup_epochs <epochs>
```
* Any-to-one:
```bash
$ python train.py --dataset <dataset name> --attack_choice any2one --attack_ratio <ratio> --aug <augment> --s <beta>
```
* Adaptive any-to-one:
```bash
$ python train_learn.py --dataset <dataset name> --attack_choice any2one --attack_ratio <ratio> --aug <augment> --s <beta> --warmup_epochs <epochs> --eps <constraint>
```

The parameter choices for the above commands are as follows:
- Dataset `<dataset name>`: `cifar10` , `celeba`, `tinyimagenet`.
- Poison proportion `<ratio>`: `0` ~ `100`
- Choice of augmentation `<augment>`: `autoaug`, `randaug`
- Trigger initialization `<beta>`: `1` , `2`, `4` `...`
- Warmup epochs `<epochs>`: `0` ~ `10`
- Learned trigger constraint boundary `<constraint>`: `0.1` (for CIFAR-10), `0.01` (for CelebA), `0.2` (for t-ImageNet)
- `--target_label`: Labels to be attacked, only in `any2one` mode.

The trained checkpoints will be saved at `checkpoints/`.

## Evaluation

To evaluate trained models, run command:

#### Any-to-any:
```bash
$ python test.py --dataset <dataset name> --attack_choice any2any --attack_ratio <ratio> --aug <augment> --s <beta>

```

#### Any-to-one:
```bash
$ python test.py --dataset <dataset name> --attack_choice any2one --attack_ratio <ratio> --aug <augment> --s <beta>
```
