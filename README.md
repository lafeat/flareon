# Flareon: Stealthy Backdoor Injection via Poisoned Augmentation

## Introdution

This is the official release 
of "Flareon: Stealthy Backdoor Injection via Poisoned Augmentation."

![image](https://github.com/lafeat/flareon/blob/main/utils/img.png)

## Requirements

- Install required python packages:
```bash
$ python -m pip install -r requirements.py
```

## Experiments
Training commands are as follows.

#### Any-to-any:
```bash
$ python train.py --dataset <datasetName> --attack_ratio <ratio> --ag <augment> --s <beta>
```
##### Adaptive attacks:
```bash
$ python train_learn.py --dataset <datasetName> --attack_ratio <ratio> --ag <augment> --s <beta> --warmup_epochs <epochset>
```
#### Any-to-one:
```bash
$ python train.py --dataset <datasetName> --attack_choice any2one --attack_ratio <ratio> --ag <augment> --s <beta>
```
##### Adaptive attacks:
```bash
$ python train_learn.py --dataset <datasetName> --attack_choice any2one --attack_ratio <ratio> --ag <augment> --s <beta> --warmup_epochs <epochset> --eps <constrain>
```
where the parameter choices are as follows:
- `<datasetName>`: `cifar10` , `celeba`, `tinyimagenet`.
- `<ratio>`: `range[0, 100]`
- `<augment>`: `flowag` , `autoag`, `randag`
- `<beat>`: `1` , `2`, `4` `...`
- `<epochset>`: `0` ~ `10`
- `<constrain>`: `0.1 (for CIFAR10)` `0.01 (for CelebA)``0.2 (for t-ImageNet)`

Other parameters are the following:
- `--target_label`: Labels to be attacked

The trained checkpoints will be saved at the path `checkpoints`.

To evaluate trained models or resume training, run command:

#### Any-to-any:
```bash
$ python test.py --dataset <datasetName> --attack_choice any2any --attack_ratio <ratio> --ag <augment> --s <beta>

```

#### Any-to-one:
```bash
$ python test.py --dataset <datasetName> --attack_choice any2one --attack_ratio <ratio> --ag <augment> --s <beta>
```

#### For adaptive learning:
```
--warmup_epochs <epochset>
```
