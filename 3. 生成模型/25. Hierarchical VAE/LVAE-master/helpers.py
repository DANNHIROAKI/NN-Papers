import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from parmesan.distributions import log_normal2, log_stdnormal
import theano.tensor as T

def plotLLs(name,outfolder,xepochs,costs,log_px,log_pz,log_qz):
    plt.figure(figsize=[12,12])
    plt.plot(xepochs,costs, label="LL")
    plt.plot(xepochs,log_px, label="logp(x|z1)")
    for ii,p in enumerate(log_pz):
        plt.plot(xepochs,p, label="log p(z%i)"%ii)
    for ii,p in enumerate(log_qz):
        plt.plot(xepochs,p, label="log q(z%i)"%ii)
    plt.xlabel('Epochs'), plt.ylabel('log()'), plt.grid('on')
    plt.ylim([-150,0])
    plt.title(name), plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(outfolder+'/'+ name +'.png'),  plt.close()

def plotLLssemisub(name,outfolder,xepochs,costs,log_px,log_pz,log_qz):
    plt.figure(figsize=[12,12])
    for ii,c in enumerate(costs):
        plt.plot(xepochs,c, label="cost-%i"%ii)
    plt.plot(xepochs,log_px, label="logp(x|z1)")
    for ii,p in enumerate(log_pz):
        plt.plot(xepochs,p, label="log p(z%i)"%ii)
    for ii,p in enumerate(log_qz):
        plt.plot(xepochs,p, label="log q(z%i)"%ii)
    plt.xlabel('Epochs'), plt.ylabel('log()'), plt.grid('on')
    plt.title(name), plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(outfolder+'/'+ name +'.png'),  plt.close()

def plotKLs(name,outfolder,xepochs,KL,vmin=0,vmax=2):
    fig, ax = plt.subplots()
    data = np.concatenate(KL,axis=0).T
    heatmap = ax.pcolor(data, cmap=plt.cm.Greys,  vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
    ax.set_xticklabels(xepochs, minor=False)
    plt.xlabel('Epochs'), plt.ylabel('#Latent Unit'), plt.title('KL(q|p)'), plt.colorbar(heatmap)
    plt.savefig(outfolder+'/' + name +'.png'),  plt.close()

def init_res(num_layers):
    res = {}
    res['cost'] = []
    res['log_px'] = []
    res['log_pz'] = [[] for i in range(num_layers)]
    res['log_qz'] = [[] for i in range(num_layers)]
    res['KL_qp'] = [[] for i in range(num_layers)]
    res['epoch'] = []
    res['acc'] = []
    return res

def init_res_test(num_layers):
    res = {}
    res['cost'] = []
    res['log_px'] = []
    res['log_pz'] = [[] for i in range(num_layers)]
    res['log_qz'] = [[] for i in range(num_layers)]
    res['KL_qp'] = [[] for i in range(num_layers)]
    res['epoch'] = []
    res['acc'] = []
    res['Neff_mean'] = []
    res['Neff_var'] = []
    return res

def add_res_test(model_out,epoch,res):
    cost, log_px, log_pz, log_qz, KL, Neff_mean, Neff_var = model_out
    num_layers = len(KL)
    res['cost'] += [cost]
    res['log_px'] += [log_px]
    res['epoch'] += [epoch]
    res['Neff_mean'] += [Neff_mean]
    res['Neff_var'] += [Neff_var]
    for i in range(num_layers):
        res['log_pz'][i] +=  [log_pz[i]]
        res['log_qz'][i] += [log_qz[i]]
        res['KL_qp'][i] += [KL[i]]
    return res


def init_res_semisub(num_layers):
    res = {}
    res['cost'] = None
    res['log_px'] = []
    res['log_pz'] = [[] for i in range(num_layers)]
    res['log_qz'] = [[] for i in range(num_layers)]
    res['KL_qp'] = [[] for i in range(num_layers)]
    res['epoch'] = []
    res['acc'] = []
    return res

def add_res(model_out,epoch,res):
    cost, log_px, log_pz, log_qz, KL = model_out
    num_layers = len(KL)
    res['cost'] += [cost]
    res['log_px'] += [log_px]
    res['epoch'] += [epoch]
    for i in range(num_layers):
        res['log_pz'][i] +=  [log_pz[i]]
        res['log_qz'][i] += [log_qz[i]]
        res['KL_qp'][i] += [KL[i]]
    return res

def add_res_semisub(model_out,epoch,res):
    cost, log_px, log_pz, log_qz, KL, confmat = model_out
    num_layers = len(KL)
    if res['cost'] is None:
        res['cost'] = [[] for _ in range(len(cost))]
    for i in range(len(cost)):
        res['cost'][i] += [cost[i]]

    res['log_px'] += [log_px]
    res['epoch'] += [epoch]
    res['acc'] += [confmat.accuracy()]
    for i in range(num_layers):
        res['log_pz'][i] +=  [log_pz[i]]
        res['log_qz'][i] += [log_qz[i]]
        res['KL_qp'][i] += [KL[i]]
    return res

def latent_gaussian_x_bernoulli(z, z_mu_q, z_logvar_q, z_mu_p, z_logvar_p, x_mu, x, eq_samples, iw_samples, latent_sizes, num_features, epsilon=1e-6,reverse_z=False,clip_val=None, temp=None, epoch=None):
    """
    Latent z       : gaussian with standard normal prior
    decoder output : bernoulli

    When the output is bernoulli then the output from the decoder
    should be sigmoid.
    z: (batch_size*nsamples*ivae_samples*nsamples, num_laten)
    x_mu: (batch_size*nsamples*ivae_samples, num_laten)
    """
    if reverse_z:
        #for ladder like VAE where x->z3->z2->z1
        z = [z_.reshape((-1, eq_samples, iw_samples, ls)) for z_,ls in zip(z,latent_sizes)]
        z_mu_q = [z_.reshape((-1, eq_samples, iw_samples, ls)) for z_,ls in zip(z_mu_q[:-1],latent_sizes[:-1])] + [z_mu_q[-1].dimshuffle((0,'x','x',1))]
        z_logvar_q = [z_.reshape((-1, eq_samples, iw_samples, ls)) for z_,ls in zip(z_logvar_q[:-1],latent_sizes[:-1])] + [z_logvar_q[-1].dimshuffle((0,'x','x',1))]
    else:
        #for normal VAE where x->z1->z2->z3
        z = [z_.reshape((-1, eq_samples, iw_samples, ls)) for z_,ls in zip(z,latent_sizes)]
        z_mu_q = [z_mu_q[0].dimshuffle((0,'x','x',1))] + [z_.reshape((-1, eq_samples, iw_samples, ls)) for z_,ls in zip(z_mu_q[1:],latent_sizes[1:])]
        z_logvar_q =  [z_logvar_q[0].dimshuffle((0,'x','x',1))] + [z_.reshape((-1, eq_samples, iw_samples, ls)) for z_,ls in zip(z_logvar_q[1:],latent_sizes[1:])]

    z_mu_p = [z_.reshape((-1, eq_samples, iw_samples, ls)) for z_,ls in zip(z_mu_p,latent_sizes[:-1])]
    z_logvar_p = [z_.reshape((-1, eq_samples, iw_samples, ls)) for z_,ls in zip(z_logvar_p,latent_sizes[:-1])]

    x_mu = x_mu.reshape((-1, eq_samples, iw_samples,  num_features))
    x = x.dimshuffle((0,'x','x',1))

    log_pz =  [log_normal2(z_, mu_, logvar_) for z_, mu_, logvar_ in zip(z[:-1],z_mu_p,z_logvar_p)] + [log_stdnormal(z[-1])]
    log_qz = [log_normal2(z_, mu_, logvar_) for z_, mu_, logvar_ in zip(z, z_mu_q,z_logvar_q)]
    log_px = -T.nnet.binary_crossentropy(T.clip(x_mu,epsilon,1-epsilon), x)

    if clip_val is not None:
        log_pz = [T.clip(lp,clip_val,0) for lp in log_pz]
        log_qz = [T.clip(lp,clip_val,0) for lp in log_qz]

    #all log_*** should have dimension (batch_size, nsamples, ivae_samples)


    nlayers = len(log_qz)

    if temp is None:
        temp = [1.0 for _ in range(nlayers)]
    else:
        temp_step = (nlayers+1)*temp/float(100)
        temp = [T.max((i+1)*temp-epoch*temp_step,0.0) for i in range(nlayers)]

    a = log_px.sum(axis=3) + sum([p.sum(axis=3)*t for p,t in zip(log_pz,temp)]) - sum([p.sum(axis=3) for p in log_qz])
    a_max = T.max(a, axis=2, keepdims=True) #(batch_size, nsamples, 1)
    #It is important that a_max is inside the mean since it is sample specific

    # T.exp(a-a_max): (bathc_size, nsamples, ivae_samples)
    # -> a_max to avoid overflow which is a problem. a_max is specific for
    # each set importance set of samples and is broadcoasted over the last dimension.
    #
    # T.log( T.mean(T.exp(a-a_max), axis=2): (bathc_size, nsamples)
    # -> This is the log of the sum over the importance weithed samples
    #
    # a_max.reshape((-1,nsamples)) (batch_size, nsamples)
    # -> We need to remove the last dimension of a_max to make the addition
    #
    # a_max.reshape((-1,nsamples)) + T.log( T.mean(T.exp(a-a_max), axis=2)) (batch_size, nsamples)
    # -> This is the LL estimater, eq (8) in Burda et. al. 2015, where nsamples is used to estimate the expectation
    # Last the LL estimator is meaned over all diemensions
    lower_bound = T.mean( a_max) + T.mean( T.log( T.mean(T.exp(a-a_max), axis=2)))
    return lower_bound, log_px, log_pz, log_qz