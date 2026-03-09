
nt = 1500
dt = 0.002
delay = 0.3
fm = 8
true_path = 'overthrust/overthrust_true.npy'
smooth_path = 'overthrust/overthrust_smooth.npy'
# smooth_path = 'marmousi_linear.npy'
dh = 25.
spatial_order = 4

abcn = 20
free_surface = False
use_habc = False

src_step = 2
rec_step = 1
srcz = 1
recz = 1

lr = 25
epochs = 101

batchsize = 8
step_per_epoch = 4
batch_per_step = int(batchsize/step_per_epoch)
show_every = 10