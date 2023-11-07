sudo systemctl stop nvidia-dcgm

while ! sudo dyno dcgm_profiling --mute=true --duration=3600_s; do
  echo "Retry"
done
echo "done"
