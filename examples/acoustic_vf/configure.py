
nt = 2500
dt = 0.002
delay = 0.256
fm = 12.
true_vp_path = '/ibex/user/wangs0j/repo/ifwi/data/models/marmousi2/true_vp.npy'
true_rx_path = '/ibex/user/wangs0j/repo/ifwi/data/models/marmousi2/true_rx.npy'
true_rz_path = '/ibex/user/wangs0j/repo/ifwi/data/models/marmousi2/true_rz.npy'
true_rho_path = '/ibex/user/wangs0j/repo/ifwi/data/models/marmousi2/true_rho.npy'
# smooth_path = 'marmousi_linear.npy'
dh = 18.75
spatial_order = 4

abcn = 50
free_surface = True
use_habc = False

src_step = 2
rec_step = 1
srcz = 0
recz = 1

lr = 25
epochs = 101

batchsize = 8
step_per_epoch = 4
batch_per_step = int(batchsize/step_per_epoch)
show_every = 10