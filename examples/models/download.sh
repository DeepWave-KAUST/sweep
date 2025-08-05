echo "Downloading Marmousi..."
save_dir=models/marmousi2
mkdir -p $save_dir
wget -P $save_dir https://s3.amazonaws.com/open.source.geoscience/open_data/elastic-marmousi/elastic-marmousi-model.tar.gz
tar -xvzf $save_dir/elastic-marmousi-model.tar.gz -C $save_dir
mkdir -p models/marmousi2/npy
tar -xvzf models/marmousi2/elastic-marmousi-model/model/MODEL_DENSITY_1.25m.segy.tar.gz -C models/marmousi2/npy
tar -xvzf models/marmousi2/elastic-marmousi-model/model/MODEL_S-WAVE_VELOCITY_1.25m.segy.tar.gz -C models/marmousi2/npy
tar -xvzf models/marmousi2/elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy.tar.gz -C models/marmousi2/npy