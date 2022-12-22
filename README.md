# affetto-nn-test

This repository contains programs to test if Affetto control can be
achieved with neural networks. As programs in this repository are
developed for testing, we cannot guarantee that they will work.

## Getting started

Clone this repository with `--recursive` option.
``` shell
git clone --recursive https://github.com/affetto-software/affetto-nn-test.git

```
Or, clone this repository and then update submodules.
``` shell
git clone https://github.com/affetto-software/affetto-nn-test.git
cd affetto-nn-test && git submodule update --init

```

As dependencies are listed in [pyproject.toml](pyproject.toml), you
need to install them with your preferred way. If you are using
[PDM](https://pdm.fming.dev/latest/), type the following to install
dependencies and create a virtual environment.
``` shell
cd affetto-nn-test  # go to affetto-nn-test directory
pdm install

```

## Usage
### Configuration file

You need a configuration file to use programs in this repository.
To create a configuration file, copy an example configuration file
from `affctrllib` directory and modify values in it if you need.
``` shell
cd affetto-nn-test  # go to affetto-nn-test directory
cp affctrllib/apps/config/config_example.toml apps/config.toml

```

Please check the following items carefully.
```
[affetto.comm]
[affetto.comm.remote]
host = "192.168.5.10"  # IP address of Affetto PC
port = 50010

[affetto.comm.local]
host = "192.168.5.109"  # IP address of your machine
port = 50000

[affetto.state]
freq = 100  # Use exact the same value in Affetto_AbstractionLayer.exe

```

### Generation of random movements

To generate random movements and save pressures at valves, pressures
in actuators and joint angles in files, use
[collect_data_random_traj.py](apps/collect_data_random_traj.py).

It can be executed, for example, with the following command:
``` shell
pdm run ./apps/collect_data_random_traj.py --config apps/config.toml \
  --output ~/data/collected_data_directory --joint 2 3 4 5 --time 120 \
  --max-diff 40 --time-range 2 4 -n 10

```
If you don't use [PDM](https://pdm.fming.dev/latest/), omit `pdm run`
and use an appropriate command to run in your virtual environment
instead.

The option `--output` specifies a path to a directory where time
series data files will be saved.

The option `--joint` specifies joint indices to be moved. Joint index
`0` is the waist joint, Joint indices `2,3,4,5` are joints in the left
shoulder and elbow.

The option `--time` specifies the time duration in seconds for each
recorded motion file, which will be saved as a file.

The option `--max-diff` specifies the maximum difference in normalized
joint angle when being updated. Normalized joint angle means a value
between 0 and 100 that the minimum and maximum values for each joint
angle are normalized to. If you set a large value the robot will make
a big move.

The option `--time-range` specifies the minimum and maximum time to
update the reference. Actual timing to update the reference will be
chosen at random from this time range. If you set small values the
robot will move quickly.

The option `-n` specifies how many motions will be generated
repeatedly in this configuration. The command above, for example, 10
file of 120 seconds will be saved in
`~/data/collected_data_directory`.

### Track a sinusoidal trajectory with trained model

One way to evaluate and compare performances of neural networks when
applying to Affetto is to track a specific trajectory with trained
models. You can use
[track_periodic_traj_mlp.py](apps/track_periodic_traj_mlp.py) for this
purpose.

It can be executed, for example, with the following command:
``` shell
pdm run ./apps/track_periodic_traj_mlp.py --config apps/config.toml
  --output ~/data/output_filename.csv \
  --train-data ~/data/collected_data_directory --n-predict 10 \
  --n-ctrl-period 1 --joint 5 --trajectory sin --time-duration 30 \
  --amplitude 40 --period 1 --ctrl mlp

```
This program will perform training of a neural network model, produce
a specific reference trajectory (sinusoidal or step), and make the
robot to track the produced reference trajectory with the trained
neural network model.

The option `--output` specifies a path to a file in which issued valve
commands and sensory data are saved.

The option `--train-data` specifies a directory that contains files
recorded by `collect_data_random_traj.py`, which will be used for
training of neural networks.

The option `--n-predict` specifies how many number of samples are
delayed in training data. There exists a delay between the valves and
the actuators due to the long trasmission lines of air. This parameter
compensates this delay. From our experiments, between 7 to 10 is best.

The option `--n-ctrl-period` is not currently used. Please set to 1.

The option `--joint` specifies the joint index to be moved.

The option `--trajectory` specifies the shape of the reference
trajectory. You can choose between `sin` and `step`. `sin` will
produce a sinusoidal trajectory, and `step` will produce a step-shape
trajectory.

The options `--amplitude` and `--period` specifies the amplitude and
the period of sinusoidal or step functions, respectively.

The option `--time-duration` specifies the time duration in seconds of
the reference trajectory.

If the option `--ctrl` is omitted or set to `mlp`, the multi-layered
perceptron will be used as the neural network model. If the option is
set to `pid`, don't train any neural network model. Instead, just try
the postion-based control, i.e. PID control, to track the produced
reference trajectory. So you can compare performances of the
multi-layered perceptron and just the PID control for the same
reference trajectory by switching this option.
