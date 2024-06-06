""" Train one step """
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.composite as C
import mindspore.ops.operations as P
import mindspore.ops.functional as F
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size


class GenWithLossCell(nn.Cell):
    """Generator with loss(wrapped)"""
    def __init__(self, netG, netD):
        super(GenWithLossCell, self).__init__()
        self.netG = netG
        self.netD = netD
        self.reduce_mode = P.ReduceMean()

    def construct(self, noise):
        """construct"""
        fake = self.netG(noise)
        d_out = self.netD(fake)
        loss_G = -self.reduce_mode(d_out)
        return (fake, loss_G)


class DisWithLossCell(nn.Cell):
    """ Discriminator with loss(wrapped) """
    def __init__(self, netG, netD):
        super(DisWithLossCell, self).__init__()
        self.netG = netG
        self.netD = netD
        self.reduce_mode = ops.ReduceMean()

    def construct(self, real, noise):
        """construct"""
        d_out_real = self.netD(real)
        fake = self.netG(noise)
        d_out_fake = self.netD(fake)
        loss_D = -(self.reduce_mode(d_out_real) - self.reduce_mode(d_out_fake))
        return loss_D


class ClipParameter(nn.Cell):
    """ Clip the parameter """
    def __init__(self):
        super(ClipParameter, self).__init__()
        self.cast = P.Cast()
        self.dtype = P.DType()

    def construct(self, params, clip_lower, clip_upper):
        """construct"""
        new_params = ()
        for param in params:
            dt = self.dtype(param)
            t = C.clip_by_value(param, self.cast(F.tuple_to_array((clip_lower,)), dt),
                                self.cast(F.tuple_to_array((clip_upper,)), dt))
            new_params = new_params + (t,)

        return new_params


class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to return generator loss.
    Args:
        network (Cell): The target network to wrap.
    """
    def __init__(self, network):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, noise):
        _, lg = self.network(noise)
        return lg



class GenTrainOneStepCell(nn.Cell):
    """ Generator TrainOneStepCell """

    def __init__(self, G_with_loss, optimizer, sens=1.0):
        super(GenTrainOneStepCell, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.G_with_loss = G_with_loss
        self.G_with_loss.set_grad()
        self.G_with_loss.set_train()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
        self.loss_net = WithLossCell(G_with_loss)
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, noise):
        fake_img, loss_G = self.G_with_loss(noise)
        sens = ops.Fill()(ops.DType()(loss_G), ops.Shape()(loss_G), self.sens)
        grads_g = self.grad(self.loss_net, self.weights)(noise, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads_g = self.grad_reducer(grads_g)

        return fake_img, ops.depend(loss_G, self.optimizer(grads_g))


_my_adam_opt = C.MultitypeFuncGraph("_my_adam_opt")


@_my_adam_opt.register("Tensor", "Tensor")
def _update_run_op(param, param_clipped):
    param_clipped = F.depend(param_clipped, F.assign(param, param_clipped))
    return param_clipped



class DisTrainOneStepCell(nn.Cell):
    """ Discriminator TrainOneStepCell """
    def __init__(self, D_with_loss, optimizer, clip_lower=-0.01, clip_upper=0.01, sens=1.0):
        super(DisTrainOneStepCell, self).__init__(auto_prefix=False)
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper
        self.optimizer = optimizer
        self.D_with_loss = D_with_loss
        self.D_with_loss.set_grad()
        self.D_with_loss.set_train()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
        self.clip_parameters = ClipParameter()
        self.hyper_map = C.HyperMap()
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, real_img, noise):
        loss_D = self.D_with_loss(real_img, noise)
        sens_d = ops.Fill()(ops.DType()(loss_D), ops.Shape()(loss_D), self.sens)
        grads_d = self.grad(self.D_with_loss, self.weights)(real_img, noise, sens_d)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads_d = self.grad_reducer(grads_d)

        upd = self.optimizer(grads_d)
        weights_D_cliped = self.clip_parameters(self.weights, self.clip_lower, self.clip_upper)
        res = self.hyper_map(F.partial(_my_adam_opt), self.weights, weights_D_cliped)
        res = F.depend(upd, res)
        return F.depend(loss_D, res)
