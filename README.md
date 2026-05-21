# L2exp

`L2exp`는 진짜 2-layer SNN 실험입니다.

구조는 다음과 같습니다.

```text
MNIST image
  -> 기존 학습 완료 625-neuron L1 SNN checkpoint
  -> L1 spike train
  -> 새로 학습하는 L2 SNN layer
  -> voting/readout test accuracy
```

L1은 새로 학습하지 않고 기존 checkpoint를 고정해서 사용합니다.
L2는 softmax/classifier가 아니라, 자체 membrane potential, inhibition, sen/vth adaptation, STDP, neuron_expect voting을 갖는 SNN layer입니다.

## 구성

```text
L2exp/
  train_l2_standalone.py
  train_l2_sen.py
  train_l2_vth.py
  README.md
  snn/
    mnist/
    l1_sen/
      random/
      model/
    l1_vth/
      random/
      model/
```

## sen/vth 병렬 실행

서버에서 두 방식을 동시에 실행:

```bash
cd ~/L2exp

nohup "$HOME/miniconda3/envs/l1snn/bin/python" -u train_l2_sen.py > server_run_l2_sen_001.log 2>&1 &
echo $! > server_run_l2_sen_001.pid

nohup "$HOME/miniconda3/envs/l1snn/bin/python" -u train_l2_vth.py > server_run_l2_vth_001.log 2>&1 &
echo $! > server_run_l2_vth_001.pid
```

기본값:

```text
train_l2_sen.py -> --modes sen --run-name server_run_l2_sen_001
train_l2_vth.py -> --modes vth --run-name server_run_l2_vth_001
```

## 전체 순차 실행

```bash
cd ~/L2exp
python train_l2_standalone.py --run-name server_run_l2_snn_001
```

## 진행 확인

```bash
tail -f server_run_l2_sen_001.log
tail -f server_run_l2_vth_001.log

ps -p $(cat server_run_l2_sen_001.pid) -o pid,etime,pcpu,pmem,cmd
ps -p $(cat server_run_l2_vth_001.pid) -o pid,etime,pcpu,pmem,cmd

nvidia-smi
```

## 결과 위치

```text
L2exp/results/l2_snn/<run_name>/<mode>/
```

mode별 주요 출력:

```text
random/
  L1_to_L2.npy
  L2_E_to_I.npy
  L2_I_to_X.npy
model/
  checkpoint.npz
  train_log.txt
  weight.npy
  neuron_expect.npy
test/
  summary.txt
  summary.json
  predictions.csv
```

## 기본 실험 조건

- L1: 기존 625-neuron checkpoint 고정
- L2: 새 625-neuron SNN layer 학습
- train iterations: 60,000
- test samples: 10,000
- checkpoint/log interval: 1,000
- readout: winner, class_mean, class_sum, class_mean_nonzero, top-k variants

## 빠른 smoke test

```bash
python train_l2_sen.py --run-name smoke_sen --n-l2 20 --train-iters 1 --test-limit 1 --save-interval 1
python train_l2_vth.py --run-name smoke_vth --n-l2 20 --train-iters 1 --test-limit 1 --save-interval 1
```

## 이어가기

같은 `--run-name`으로 다시 실행하면 `model/checkpoint.npz`에서 이어갑니다.

처음부터 다시 하려면:

```bash
python train_l2_sen.py --force
python train_l2_vth.py --force
```
