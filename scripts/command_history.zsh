######################################
# Leaning motion of one joint
######################################

### Collecting data

## time range: 0.1-0.5
joints=(5); T=120; trange=(0.1 0.5); N=10; app=collect_data_random_traj; output="$HOME/Dropbox/work/data/affetto_nn_test/${app}/joints-${(j:-:)joints}_T-${T}_trange-${(j:-:)trange}/$(date '+%Y%m%d_%H%M%S')"; pdm run python "./apps/${app}.py" -c apps/config.toml -o "$output" -j ${joints[@]} -T ${T} -t ${trange[@]} -n ${N}; notify-send -u critical "Done."

## time range: 1-2
joints=(5); T=120; trange=(1 2); N=10; app=collect_data_random_traj; output="$HOME/Dropbox/work/data/affetto_nn_test/${app}/joints-${(j:-:)joints}_T-${T}_trange-${(j:-:)trange}/$(date '+%Y%m%d_%H%M%S')"; pdm run python "./apps/${app}.py" -c apps/config.toml -o "$output" -j ${joints[@]} -T ${T} -t ${trange[@]} -n ${N}; notify-send -u critical "Done."

## time range: 2-4
joints=(5); T=120; trange=(2 4); N=10; app=collect_data_random_traj; output="$HOME/Dropbox/work/data/affetto_nn_test/${app}/joints-${(j:-:)joints}_T-${T}_trange-${(j:-:)trange}/$(date '+%Y%m%d_%H%M%S')"; pdm run python "./apps/${app}.py" -c apps/config.toml -o "$output" -j ${joints[@]} -T ${T} -t ${trange[@]} -n ${N}; notify-send -u critical "Done."

### Tracking periodic trajectory with PID control
joints=(5); T=30; app=track_periodic_traj_mlp; now=$(date '+%Y%m%d_%H%M%S'); ctrl=pid; traj=sin; amp=40; period=5; bias=50;
output="$HOME/Dropbox/work/data/affetto_nn_test/${app}/joints-${(j:-:)joints}_${traj}_a-${amp}_p-${period}_b-${bias}/${ctrl}";
pdm run "./apps/${app}.py" -c apps/config.toml -o "${output}/${now}.csv" --ctrl "$ctrl" -j ${joints[@]} -t "$traj" -D $T -a $amp -p $period -b $bias;

data="${output}/${now}.csv"
[[ -f "$data" ]] && python3 ./apps/scripts/plot_track_traj.py "$data" --joint ${joints[@]} -d "${data%.*}" -s -x

### Learning and tracking periodic trajectory
train_data=(
"$HOME/Dropbox/work/data/affetto_nn_test/collect_data_random_traj/joints-5_T-120_trange-0.1-0.5/20240523_135438"
"$HOME/Dropbox/work/data/affetto_nn_test/collect_data_random_traj/joints-5_T-120_trange-1-2/20240523_142010"
"$HOME/Dropbox/work/data/affetto_nn_test/collect_data_random_traj/joints-5_T-120_trange-2-4/20240523_145644"
);
n_predict=10; scaler=minmax;
joints=(5); T=30; app=track_periodic_traj_mlp; now=$(date '+%Y%m%d_%H%M%S'); ctrl=mlp; traj=sin; amp=40; period=5; bias=50;
output="$HOME/Dropbox/work/data/affetto_nn_test/${app}/joints-${(j:-:)joints}_${traj}_a-${amp}_p-${period}_b-${bias}/${ctrl}_${scaler}_${n_predict}";
pdm run "./apps/${app}.py" -c apps/config.toml -o "${output}/${now}.csv" --ctrl "$ctrl" -j ${joints[@]} -t "$traj" -D $T -a $amp -p $period -b $bias --train-data "${train_data[@]}" --n-predict $n_predict --n-ctrl-period 1 --scaler $scaler;
[[ -f model.joblib ]] && mv model.joblib "${output}/${now}.joblib"

data="${output}/${now}.csv"
[[ -f "$data" ]] && python3 ./apps/scripts/plot_track_traj.py "$data" --joint ${joints[@]} -d "${data%.*}" -s -x

### Recording reference trajectory
cd ~/develop/affctrllib
mkdir -p "$HOME/Dropbox/work/data/affetto_nn_test/filter_sensory_data"

joints=(5); T=30; app=filter_sensory_data; output="$HOME/Dropbox/work/data/affetto_nn_test/${app}/joints-${(j:-:)joints}_$(date '+%Y%m%d_%H%M%S').csv"; pdm run python "./apps/${app}.py" -c apps/config.toml -o "$output" -T ${T}

### Tracking recorded trajectory with PID control
#record="$HOME/Dropbox/work/data/affetto_nn_test/filter_sensory_data/joints-5_20240523_161044.csv"
#record="$HOME/Dropbox/work/data/affetto_nn_test/filter_sensory_data/joints-5_20240524_124522.csv"
record="$HOME/Dropbox/work/data/affetto_nn_test/track_recorded_traj_mlp/joints-5_20240524_124522/mlp_minmax_10/20240524_124819.csv"
joints=(5); T=30;

app=track_recorded_traj_mlp; now=$(date '+%Y%m%d_%H%M%S'); ctrl=pid;
output="$HOME/Dropbox/work/data/affetto_nn_test/${app}/${${record%.*}##*/}/${ctrl}";
pdm run "./apps/${app}.py" -c apps/config.toml -o "${output}/${now}.csv" --ctrl "$ctrl" -j ${joints[@]} -D $T --record-data "${record}";

data="${output}/${now}.csv"
[[ -f "$data" ]] && python3 ./apps/scripts/plot_track_traj.py "$data" --joint ${joints[@]} -d "${data%.*}" -s -x

### Learning and tracking recorded trajectory
train_data=(
"$HOME/Dropbox/work/data/affetto_nn_test/collect_data_random_traj/joints-5_T-120_trange-0.1-0.5/20240523_135438"
"$HOME/Dropbox/work/data/affetto_nn_test/collect_data_random_traj/joints-5_T-120_trange-1-2/20240523_142010"
"$HOME/Dropbox/work/data/affetto_nn_test/collect_data_random_traj/joints-5_T-120_trange-2-4/20240523_145644"
);
n_predict=10; scaler=minmax;
app=track_recorded_traj_mlp; now=$(date '+%Y%m%d_%H%M%S'); ctrl=mlp;
output="$HOME/Dropbox/work/data/affetto_nn_test/${app}/${${record%.*}##*/}/${ctrl}_${scaler}_${n_predict}";
pdm run "./apps/${app}.py" -c apps/config.toml -o "${output}/${now}.csv" --ctrl "$ctrl" -j ${joints[@]} -D $T --record-data "${record}" --train-data "${train_data[@]}" --n-predict $n_predict --n-ctrl-period 1 --scaler $scaler;
[[ -f model.joblib ]] && mv model.joblib "${output}/${now}.joblib"

data="${output}/${now}.csv"
[[ -f "$data" ]] && python3 ./apps/scripts/plot_track_traj.py "$data" --joint ${joints[@]} -d "${data%.*}" -s -x
