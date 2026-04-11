python acoustic_efficiency_compare.py   --dim 2d   --checkpoint-chunks 50,100,200 --nshots 8 --checkpoint-counts 2,8,64  --boundary-storages gpu,cpu   --transfer-intervals 1,2,4,8   --warmup 3   --repeats 10  --include-deepwave

python acoustic_efficiency_compare.py   --dim 3d   --checkpoint-chunks 50,100,200 --nshots 2 --checkpoint-counts 2,8,64  --boundary-storages gpu,cpu   --transfer-intervals 1,2,4,8   --warmup 3   --repeats 10  --include-deepwave

python elastic_efficiency_compare.py   --dim 2d   --checkpoint-chunks 50,100,200 --nshots 2 --checkpoint-counts 2,8,64  --boundary-storages gpu,cpu   --transfer-intervals 1,2,4,8   --warmup 3   --repeats 10