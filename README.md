# Flareon: Stealthy Backdoor Injection via Poisoned Augmentation

## Introdution

This is the official repository
of "Flareon: Stealthy Backdoor Injection via Poisoned Augmentation."

<img src="https://github.com/lafeat/flareon/blob/main/asset/overview.png" width="700px">


## Requirements

- Install required python packages:
```shell
python -m pip install -r requirements.py
```

## Training
Training commands are as follows.

* Any-to-any:
``shell
python train.py \
    --dataset <dataset name> \
    --attack_ratio <ratio>   \
    --aug <augment>          \
    --s <beta>
```

* Adaptive any-to-any:
```shell
python train.py \
    --dataset <dataset name> \
    --attack_ratio <ratio>   \
    --aug <augment>          \
    --s <beta>               \
    --warmup_epochs <epochs> \
    --eps <constraint>
```

* Any-to-one:
```shell
python train.py \
    --dataset <dataset name> \
    --attack_choice any2one  \
    --attack_ratio <ratio>   \
    --aug <augment>          \
    --s <beta>
```

* Adaptive any-to-one:
```shell
python train.py \
    --dataset <dataset name> \
    --attack_choice any2one  \
    --attack_ratio <ratio>   \
    --aug <augment>          \
    --s <beta>               \
    --warmup_epochs <epochs> \
    --eps <constraint>
```


The parameter choices for the above commands are as follows:
- Dataset `<dataset name>`: `cifar10` , `celeba`, `tinyimagenet`.
- Poisoned proportion per batch `<ratio>`: `0` ~ `100`
- Choice of augmentation `<augment>`: `autoaug`, `randaug`
- Trigger initialization `<beta>`: `1` , `2`, `4`, `...`
- Warmup epochs `<epochs>`: `0` ~ `10`
- Learned trigger constraint boundary `<constraint>`:
  `0.1` (for CIFAR-10), `0.01` (for CelebA), `0.2` (for t-ImageNet)

The trained checkpoints will be saved at `checkpoints/`.

## Evaluation

To evaluate trained models, run command:

#### Any-to-any:
```shell
python test.py \
    --dataset <dataset name>  \
    --attack_ratio <ratio>   \
    --attack_choice any2any  \
    --s <beta>
```

#### Any-to-one:
```shell
python test.py \
    --dataset <dataset name>  \
    --attack_ratio <ratio>   \
    --attack_choice any2one  \
    --s <beta>
```

## Acknowledgement
- Code adapted from the official implementation of WaNet:
  [[Code]](https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release)
  [[Paper]](https://openreview.net/pdf?id=eEn8KTtJOx).

