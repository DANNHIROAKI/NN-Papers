import cProfile, pstats, io
import torch
from sru import SRUCell

def test_fwd_cpu():
    cell = SRUCell(3, 5, use_tanh=True)
    mask = torch.zeros(7, 1)
    mask[0,0]=1
    mask[6,0]=1
    x = torch.randn(7, 1, 3)
    with torch.no_grad():
        out_1 = cell(x, mask_pad=mask)
    out_2 = cell(x, mask_pad=mask)
    print("----------")
    print("CPU fwd optimized: {} {}".format(out_1[0].sum(), out_1[1].sum()))
    print("CPU fwd unoptimized: {} {}".format(out_2[0].sum(), out_2[1].sum()))
    print("----------")
    print("")

def test_bi_fwd_cpu():
    cell = SRUCell(5, 5, bidirectional=True)
    x = torch.randn(7, 1, 5)
    mask = torch.zeros(7, 1)
    mask[0,0]=1
    mask[6,0]=1
    with torch.no_grad():
        out_1 = cell(x)
    out_2 = cell(x)
    print("----------")
    print("CPU bi-fwd optimized: {} {}".format(out_1[0].sum(), out_1[1].sum()))
    print("CPU bi-fwd unoptimized: {} {}".format(out_2[0].sum(), out_2[1].sum()))
    print("----------")
    print("")

def profile_speed():
    bcell = SRUCell(400, 200, bidirectional=True)
    bcell.eval()
    mask = torch.zeros(200, 1)
    x = torch.randn(200, 1, 400)
    pr = cProfile.Profile()
    pr.enable()
    with torch.no_grad():
        for i in range(10):
             r = bcell(x, mask_pad=mask)
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    pr = cProfile.Profile()
    pr.enable()
    #with torch.no_grad():
    for i in range(10):
        r = bcell(x, mask_pad=mask)
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

def test_custom_v(gpu=False):
    x = torch.randn(10, 2, 5)
    cell_1 = SRUCell(5, 5, bidirectional=True)
    if gpu:
        x = x.cuda()
        cell_1 = cell_1.cuda()
    weight = cell_1.weight
    bias = cell_1.bias
    weight_c = cell_1.weight_c
    h_1, c_1 = cell_1(x)
    loss_1 = h_1.sum()
    loss_1.backward()
    print("----------")
    print("SRU w/o custom_m:")
    print("loss: {}".format(loss_1))
    print("c: {}".format(c_1.sum()))
    print("grad w: {}".format(weight.grad.sum()))
    print("grad v: {}".format(weight_c.grad.sum()))
    print("")

    cell_1.zero_grad()
    weight_c_custom = weight_c.view(2,-1).transpose(0, 1).contiguous().view(-1)
    # weight_c is (2, bidir, d)
    # but custom weight_c is providing (length, batch, bidir, d, 2)
    def custom_m(input, **kwargs):
        U = input.matmul(weight)
        V = input.new_zeros(input.size(0), input.size(1), weight_c_custom.size(0))
        return U, V

    cell_2 = SRUCell(5, 5, bidirectional=True,
            custom_m=custom_m
    )
    cell_2.weight_c.data.copy_(weight_c_custom)
    cell_2.bias = bias
    if gpu:
        cell_2 = cell_2.cuda()
    h_2, c_2 = cell_2(x)
    loss_2 = h_2.sum()
    loss_2.backward()
    print("SRU w/ custom_m:")
    print("loss: {}".format(loss_2))
    print("c: {}".format(c_2.sum()))
    print("grad w: {}".format(weight.grad.sum()))
    print("grad v: {}".format(cell_2.weight_c.grad.sum()))
    print("----------")
    print("")

test_fwd_cpu()
test_bi_fwd_cpu()
test_custom_v(gpu=False)
#test_custom_v(gpu=True)
#profile_speed()

