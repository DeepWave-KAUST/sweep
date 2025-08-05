
nt = 2500
dt = 0.002
delay = 0.256
fm = 5
true_path = '../models/models/marmousi2/npy/MODEL_P-WAVE_VELOCITY_1.25m_1.25m.npy'
# smooth_path = 'marmousi_linear.npy'
spatial_order = 8

abcn = 50
free_surface = True
use_habc = False

src_step = 2
rec_step = 1
srcz = 1
recz = 1

lr = 25
epochs = 101

batchsize = 16
step_per_epoch = 4
batch_per_step = int(batchsize/step_per_epoch)
show_every = 10