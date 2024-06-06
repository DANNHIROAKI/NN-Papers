# Code used to run the experiments in
# Soenderby, C.K., Raiko, T., Maaloe, L., Sønderby, S.K. and Winther, O., 2016.
# How to Train Deep Variational Autoencoders and Probabilistic Ladder Networks.
# arXiv preprint arXiv:1602.02282.

# LICENSE
# The MIT License (MIT)
# Copyright (c) 2016 Casper Kaae Soenderby

# Permission is hereby granted, free of charge, to any person obtaining a copy of this 
# software and associated documentation files (the "Software"), to deal in the Software 
# without restriction, including without limitation the rights to use, copy, modify, 
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies 
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.
import theano
theano.config.floatX = 'float32'
import matplotlib
matplotlib.use('Agg')
import theano.tensor as T
import numpy as np
import lasagne
from parmesan.distributions import log_stdnormal, log_normal2
from parmesan.layers import NormalizeLayer, ScaleAndShiftLayer, ListIndexLayer
from parmesan.datasets import load_mnist_realval, load_omniglot, load_omniglot_iwae, load_norb_small
import matplotlib.pyplot as plt
import shutil, gzip, os, cPickle, time, operator, argparse
from helpers import plotKLs, init_res, add_res
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from sklearn.decomposition import PCA
import math


def one_hot(x,n_labels=None):
    if n_labels == None:
        n_labels = np.max(x)+1
    return np.eye(n_labels)[x]

class SampleLayer(lasagne.layers.MergeLayer):
    """
    Sampling layer supporting importance sampling as described in [BURDA]_ and
    multiple Monte Carlo samples for the approximation of
    E_q [log( p(x,z) / q(z|x) )].

    Parameters
    ----------
    mu, log_var : :class:`Layer` instances
        Parameterizing the mean and log(variance) of the distribution to sample
        from as described in [BURDA]_. The code assumes that these have the same
        number of dimensions.

    eq_samples : int or T.scalar
        Number of Monte Carlo samples used to estimate the expectation over
        q(z|x) in eq. (8) in [BURDA]_.

    iw_samples : int or T.scalar
        Number of importance samples in the sum over k in eq. (8) in [BURDA]_.

    References
    ----------
        ..  [BURDA] Burda, Yuri, Roger Grosse, and Ruslan Salakhutdinov.
            "Importance Weighted Autoencoders."
            arXiv preprint arXiv:1509.00519 (2015).
    """

    def __init__(self, mu, var, eq_samples=1, iw_samples=1, **kwargs):
        super(SampleLayer, self).__init__([mu, var], **kwargs)

        self.eq_samples = eq_samples
        self.iw_samples = iw_samples

        self._srng = RandomStreams(
            lasagne.random.get_rng().randint(1, 2147462579))

    def get_output_shape_for(self, input_shapes):
        batch_size, num_latent = input_shapes[0]
        if isinstance(batch_size, int) and \
           isinstance(self.iw_samples, int) and \
           isinstance(self.eq_samples, int):
            out_dim = (batch_size*self.eq_samples*self.iw_samples, num_latent)
        else:
            out_dim = (None, num_latent)
        return out_dim

    def get_output_for(self, input, **kwargs):
        mu, var = input
        batch_size, num_latent = mu.shape
        eps = self._srng.normal(
            [batch_size, self.eq_samples, self.iw_samples, num_latent],
             dtype=theano.config.floatX)

        z = mu.dimshuffle(0,'x','x',1) + \
                T.sqrt(var).dimshuffle(0,'x','x',1) * eps

        return z.reshape((-1,num_latent))

class LadderMergeLayer(lasagne.layers.MergeLayer):
    def __init__(self, mu_t, var_t, mu_l, var_l,  eq_samples, iw_samples, **kwargs):
        super(LadderMergeLayer, self).__init__([mu_t, var_t, mu_l, var_l], **kwargs)

        self.eq_samples =  eq_samples
        self.iw_samples = iw_samples

        self.num_inputs = self.input_shapes[0][-1]

    def get_output_shape_for(self, input_shapes):
        output_shape = list(input_shapes[0])  # make a mutable copy
        return tuple(output_shape)

    def get_output_for(self, inputs, **kwargs):
        mu_t, var_t, mu_l, var_l = inputs

        mu_t = mu_t.reshape((-1,self.eq_samples, self.iw_samples,self.num_inputs))
        var_t = var_t.reshape((-1,self.eq_samples, self.iw_samples,self.num_inputs))
        mu_l = mu_l.reshape((-1,1,1,self.num_inputs))
        var_l = var_l.reshape((-1,1,1,self.num_inputs))

        prec_t = var_t**(-1)
        prec_l = var_l**(-1)

        mu_est =  (mu_t*prec_t + mu_l*prec_l) / (prec_l+prec_t)
        var_est = ( prec_t + prec_l)**(-1)

        return mu_est.reshape((-1,self.num_inputs)), var_est.reshape((-1,self.num_inputs))

class DecoderSampleLayer(lasagne.layers.MergeLayer):
    """
    """

    def __init__(self, z_enc, mu, var, eq_samples=1, iw_samples=1, **kwargs):
        super(DecoderSampleLayer, self).__init__([z_enc, mu, var], **kwargs)

        self.eq_samples = eq_samples
        self.iw_samples = iw_samples

        self._srng = RandomStreams(
            lasagne.random.get_rng().randint(1, 2147462579))

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, drawdecsample=False, **kwargs):

        z_enc, mu, var = input

        if drawdecsample:
            batch_size, num_latent = mu.shape
            eps = self._srng.normal(
                [batch_size, self.eq_samples, self.iw_samples, num_latent],
                 dtype=theano.config.floatX)

            z = mu.dimshuffle(0,'x','x',1) + \
                 T.sqrt(var).dimshuffle(0,'x','x',1) * eps

            return z.reshape((-1,num_latent))
        else:
            return z_enc


def negsemilogy(ax,x,y,label):
    Y = np.sign(y)*np.log10(np.abs(y))
    ax.plot(x,Y,label=label)


def plotLLs(name,outfolder,xepochs,costs,log_px,log_pz,log_qz):
    fig, ax = plt.subplots()
    negsemilogy(ax,xepochs,costs, label="LL")
    negsemilogy(ax,xepochs,log_px, label="logp(x|z1)")
    for ii,p in enumerate(log_pz):
        negsemilogy(ax,xepochs,p, label="log p(z%i)"%ii)
    for ii,p in enumerate(log_qz):
        negsemilogy(ax,xepochs,p, label="log q(z%i)"%ii)
    plt.xlabel('Epochs'), plt.ylabel('log()'), ax.grid('on')
    plt.ylim([-4,4])

    ax.set_yticks([-4,-3,-2,-1,0,1,2,3,4])
    ax.set_yticklabels(["-1e4","-1e3","-1e2","-1e1","-1/1","1e1","1e2","1e3","1e4"])
    plt.title(name), plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(outfolder+'/'+ name +'.png'),  plt.close()


def plotsamples(name,outfolder,samples):
    shp = samples.shape[1:]
    nsamples = samples.shape[0]

    samples_pr_size = int(np.sqrt(nsamples))
    if len(shp) == 3:
        canvas = np.zeros((h*samples_pr_size, samples_pr_size*w,shp[2]))
        cm = None
    else:
        canvas = np.zeros((h*samples_pr_size, samples_pr_size*w))
        cm = plt.gray()
    idx = 0
    for i in range(samples_pr_size):
        for j in range(samples_pr_size):
            canvas[i*h:(i+1)*h, j*w:(j+1)*w] = np.clip(samples[idx],1e-6,1-1e-6)
            idx += 1
    plt.figure(figsize=(7, 7))
    plt.imshow(canvas,cmap=cm)
    plt.savefig(outfolder+'/' + name +'.png')

def boxplot(res_out,name,data):
    #name = 'boxplot:mu_q_iw%i'%j
    #data = var_p[0]
    bs,eq,iw,nf = data.shape
    data_pl = [data.mean(axis=(1,2))[:,n] for n in range(nf)]
    fig, ax = plt.subplots()
    plt.boxplot(data_pl)
    plt.ylim([-12,12])
    plt.xlabel('Unit (min/max)')
    plt.grid('on')
    ticks = ["%0.0e/%0.0e"%(d.min(),d.max()) for d in data_pl]
    plt.xticks(range(1,len(data_pl)+1),ticks,fontsize = 5)
    plt.savefig(res_out+'/' + name +'.png')

def plotPCA(outfolder,name,z,target=None):
    fig, ax = plt.subplots()
    pca = PCA(n_components=2)
    X_r = pca.fit(z).transform(z)
    plt.figure()
    if target is not None:
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, num_class))
        for i,c in zip(range(num_class),colors):
            plt.scatter(X_r[target == i, 0], X_r[target == i, 1], c=c, label=str(i),s=5,alpha=.5)
    else:
        plt.scatter(X_r[:, 0], X_r[:, 1],s=5)
    plt.xlabel('PCA1'), plt.ylabel('PCA2'), ax.grid('on')
    #plt.ylim([-4,4])
    plt.title(name), plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(outfolder+'/'+ name +'.png'),  plt.close()



parser = argparse.ArgumentParser()
parser.add_argument("-eq_samples", type=int,
        help="Eq_samples", default=1)
parser.add_argument("-iw_samples", type=int,
        help="Iw_samples", default=1)
parser.add_argument("-lr", type=float,
        help="lr", default=0.002)
parser.add_argument("-outfolder", type=str,
        help="outfolder", default="outfolder")
parser.add_argument("-nonlin_enc", type=str,
        help="nonlin_enc", default="leaky_rectify")
parser.add_argument("-nonlin_dec", type=str,
        help="nonlin_dec", default="leaky_rectify")
parser.add_argument("-mlp_layers", type=int,
        help="mlp_layers", default=2)
parser.add_argument("-batch_size", type=int,
        help="batch_size train", default=256)
parser.add_argument("-batch_size_test", type=int,
        help="batch_size test", default=25)
parser.add_argument("-temp_start", type=float,
        help="temp_start", default=1.0)
parser.add_argument("-temp_epochs", type=int,
        help="temp_epochs", default=100)
parser.add_argument("-batch_norm", type=str,
        help="batch normalization", default='True')
parser.add_argument("-batch_norm_output", type=str,
        help="batch_norm_output", default='False')
parser.add_argument("-latent_sizes", type=str,
        help="latent_sizes", default="64,32")
parser.add_argument("-hidden_sizes", type=str,
        help="hidden_sizes", default="512,256")
parser.add_argument("-verbose", type=str,
        help="verbose printing", default="False")
parser.add_argument("-ramp_n_samples", type=str,
        help="ramp_n_samples", default="False")
parser.add_argument("-num_epochs", type=int,
        help="num_epochs", default=5000)
parser.add_argument("-eval_epochs", type=str,
        help="eval_epochs", default='1,10,100')
parser.add_argument("-dataset", type=str,
        help="mnistresample|omniglot_iwae|norb_small", default='mnistresample')
parser.add_argument("-only_mu_up", type=str,
        help="only_mu_up (only applies to ladder)", default="True")
parser.add_argument("-modeltype", type=str,
        help="ladderVAE|VAE", default='VAE')
parser.add_argument("-L2", type=float,
        help="L2", default=0.0)
parser.add_argument("-ladder_share_params", type=str,
        help="ladder_share_params (only applies to ladder)", default="False")
parser.add_argument("-lv_eps_z", type=float,
        help="small constant added to z-variance to avoid underflow", default=1e-5)
parser.add_argument("-lv_eps_out", type=float,
        help="small constant added to x-variance to avoid underflow", default=1e-5)


args = parser.parse_args()

def get_nonlin(nonlin):
    if nonlin == 'rectify':
        return lasagne.nonlinearities.rectify
    elif nonlin == 'leaky_rectify':
        return lasagne.nonlinearities.leaky_rectify
    elif nonlin == 'very_leaky_rectify':
        return lasagne.nonlinearities.very_leaky_rectify
    elif nonlin == 'tanh':
        return lasagne.nonlinearities.tanh
    elif nonlin == 'capped_leaky_rectify':
        return lambda x: T.clip(lasagne.nonlinearities.leaky_rectify(x),-0.01*2,2)
    else:
        raise ValueError()

#dump settings to the logfile
args_dict = vars(args)
sorted_args = sorted(args_dict.items(), key=operator.itemgetter(0))
description = []
description.append('######################################################')
description.append('# --Commandline Params--')
for name, val in sorted_args:
    description.append("# " + name + ":\t" + str(val))
description.append('######################################################')


iw_samples = args.iw_samples    #number of MC samples over the expectation over E_q(z|x)
eq_samples = args.eq_samples            #number of importance weighted samples
lr = args.lr
res_out = args.outfolder
nonlin_enc = get_nonlin(args.nonlin_enc)
nonlin_dec = get_nonlin(args.nonlin_dec)
batch_size = args.batch_size
batch_norm = args.batch_norm == 'True' or args.batch_norm == 'true'
batch_norm_output = args.batch_norm_output == 'True' or args.batch_norm_output == 'true'
num_mlp_layers  = args.mlp_layers
if args.eval_epochs == 'default':
    eval_epochs = [50,100,200,500,750] + range(1000,10000,500)
else:
    eval_epochs = map(int,args.eval_epochs.split(','))
verbose = args.verbose == 'True' or args.verbose == 'true'
latent_sizes = map(int,args.latent_sizes.split(','))
hidden_sizes = map(int,args.hidden_sizes.split(','))
ramp_n_samples = args.ramp_n_samples == 'True' or args.ramp_n_samples == 'true'
num_epochs = args.num_epochs
dataset = args.dataset
only_mu_up = args.only_mu_up == 'True' or args.only_mu_up == 'true'
modeltype = args.modeltype
batch_size_test = args.batch_size_test
temp_epochs = args.temp_epochs
temp_start = args.temp_start
L2  = args.L2
ladder_share_params = args.ladder_share_params == 'True' or args.ladder_share_params == 'true'
lv_eps_z = args.lv_eps_z
lv_eps_out = args.lv_eps_out

def verbose_print(text):
    if verbose: print text

w_init_mu = lasagne.init.GlorotNormal(1.0)
b_init_var = lasagne.init.Constant(1.0)
w_init_var = lasagne.init.GlorotNormal(1.0)
w_init_sigmoid = lasagne.init.GlorotNormal(1.0)
w_init_mlp = lasagne.init.GlorotNormal('relu')

if not os.path.exists(res_out):
    os.makedirs(res_out)

#dump script in result dir
scriptpath = os.path.realpath(__file__)
filename = os.path.basename(scriptpath)
shutil.copy(scriptpath,res_out + '/' + filename)
logfile = res_out + '/logfile.log'
trainlogfile = res_out + '/trainlogfile.log'
model_out = res_out + '/model'
with open(logfile,'w') as f:
    for l in description:
        f.write(l + '\n')


sym_iw_samples = T.iscalar('iw_samples')
sym_eq_samples = T.iscalar('eq_samples')
sym_lr = T.scalar('lr')
sym_x = T.matrix()
sym_mu = T.matrix()
sym_var = T.matrix()
sym_temp = T.scalar()


desc = ""
test_t = None

def bernoullisample(x):
    return np.random.binomial(1,x,size=x.shape).astype(theano.config.floatX)

#load dataset
regularize_var = False
if dataset == 'mnistresample':
    drawsamples = True
    print "Using resampled mnist dataset"
    process_data = bernoullisample
    train_x, train_t, valid_x, valid_t, test_x, test_t = load_mnist_realval()
    train_x = np.concatenate([train_x,valid_x])
    test_x = process_data(test_x)
    idx = np.random.permutation(test_x.shape[0])
    test_x = test_x[idx]
    test_t = test_t[idx]
    pcaplot = True
    num_class = 10
    h,w = 28,28
    ntrain = train_x.shape[0]
    ntest = train_x.shape[0]
    num_features = h*w
    outputdensity = 'bernoulli'
    outputnonlin = lasagne.nonlinearities.sigmoid
    imgshp = [h,w]
elif dataset == 'omniglot':
    drawsamples = True
    print "Using omniglot dataset"
    train_x, test_x = load_omniglot()
    np.random.shuffle(train_x)
    np.random.shuffle(test_x)
    process_data = bernoullisample
    h,w = 32,32
    pcaplot = True
    ntrain = train_x.shape[0]
    ntest = test_x.shape[0]
    num_features = h*w
    train_x = train_x.reshape(-1,num_features)
    test_x = test_x.reshape(-1,num_features)
    outputdensity = 'bernoulli'
    outputnonlin = lasagne.nonlinearities.sigmoid
    imgshp = [h,w]
elif dataset == 'omniglot_iwae':
    drawsamples = True
    print "Using omniglot dataset"
    train_x, train_t, train_char, test_x, test_t, test_char = load_omniglot_iwae()
    np.random.shuffle(train_x)
    np.random.shuffle(test_x)
    process_data = bernoullisample
    num_class = 50
    h,w = 28,28
    pcaplot = True
    ntrain = train_x.shape[0]
    ntest = test_x.shape[0]
    num_features = h*w
    train_x = train_x.reshape(-1,num_features)
    test_x = test_x.reshape(-1,num_features)
    outputdensity = 'bernoulli'
    outputnonlin = lasagne.nonlinearities.sigmoid
    imgshp = [h,w]
elif dataset == 'norb_small':
    print "Using norb_small dataset"
    process_data = lambda x: x
    train_x, train_t, test_x, test_t = load_norb_small(normalize=True,dequantify=True)
    ntrain = train_x.shape[0]
    ntest = train_x.shape[0]
    h,w = 32,32
    num_features = h*w
    pcaplot = True
    num_class = 5
    outputdensity = 'gaussian'
    outputnonlin = lasagne.nonlinearities.linear
    imgshp = [h,w]
    drawsamples = True
else:
    raise ValueError()

def get_mu_var(inputs):
    mu, var = ListIndexLayer(inputs,index=0),ListIndexLayer(inputs,index=1)
    return mu, var

def batchnormlayer(l,num_units, nonlinearity, name, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.)):
    l = lasagne.layers.DenseLayer(l, num_units=num_units, name="Dense-" + name, W=W, b=b, nonlinearity=None)
    l = NormalizeLayer(l,name="BN-" + name)
    l = ScaleAndShiftLayer(l,name="SaS-" + name)
    l = lasagne.layers.NonlinearityLayer(l,nonlinearity=nonlinearity,name="Nonlin-" + name)
    return l

def normaldenselayer(l,num_units, nonlinearity, name, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.)):
    l = lasagne.layers.DenseLayer(l, num_units=num_units, name="Dense-" + name, W=W, b=b, nonlinearity=nonlinearity)
    return l

if batch_norm:
    print "Using batch Normalization - The current implementation calculates " \
          "the BN constants on the complete dataset in one batch. This might " \
          "cause memory problems on some GFX's"
    denselayer = batchnormlayer
else:
    denselayer = normaldenselayer


def mlp(l,num_units, nonlinearity, name, num_mlp_layers=1, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),):
    outputlayer = l
    for i in range(num_mlp_layers):
        outputlayer = denselayer(outputlayer, num_units=num_units, name=name+'_'+str(i+1), nonlinearity=nonlinearity, W=W, b=b)
    return outputlayer


test_x = process_data(test_x)
X = process_data(train_x)[:batch_size]
num_layers = len(hidden_sizes)

if drawsamples:
    plotsamples('samples_conditioned_true',res_out,test_x[:10**2].reshape([-1]+imgshp))
    plotsamples('train_samples',res_out,train_x[:10**2].reshape([-1]+imgshp))

if modeltype == 'VAE':
    reversed_z = False
    lenc_z_mu = [[] for i in range(num_layers)]
    lenc_z_var = [[] for i in range(num_layers)]
    l_z = [[] for i in range(num_layers)]
    lenc_z_mu = [[] for i in range(num_layers)]
    lenc_z_var = [[] for i in range(num_layers)]
    lenc_zt_mu = [[] for i in range(num_layers)]
    lenc_zt_var = [[] for i in range(num_layers)]
    ldec_z_mu = [[] for i in range(num_layers)]
    ldec_z_var = [[] for i in range(num_layers)]


    #RECOGNITION MODEL
    l_in = lasagne.layers.InputLayer((None, num_features))
    l_enc_h= mlp(l_in, num_units=hidden_sizes[0], W=w_init_mlp, name='ENC_A_DENSE%i'%0, nonlinearity=nonlin_enc, num_mlp_layers=num_mlp_layers)
    lenc_z_mu[0] = denselayer(l_enc_h, num_units=latent_sizes[0], W=w_init_mu, nonlinearity=lasagne.nonlinearities.identity, name='ENC_A_MU%i'%0)
    lenc_z_var[0] = denselayer(l_enc_h, num_units=latent_sizes[0], W=w_init_var, nonlinearity=lasagne.nonlinearities.softplus, b=b_init_var, name='ENC_A_var%i'%0)
    l_z[0] = SampleLayer(mu=lenc_z_mu[0], var=lenc_z_var[0], eq_samples=sym_eq_samples, iw_samples=sym_iw_samples)

    for i in range(1,num_layers):
        l_enc_h= mlp(l_z[i-1], num_units=hidden_sizes[i], W=w_init_mlp, name='ENC_A_DENSE%i'%i, nonlinearity=nonlin_enc, num_mlp_layers=num_mlp_layers)
        lenc_z_mu[i] = denselayer(l_enc_h, num_units=latent_sizes[i], W=w_init_mu, nonlinearity=lasagne.nonlinearities.identity, name='ENC_A_MU%i'%i)
        lenc_z_var[i] = denselayer(l_enc_h, num_units=latent_sizes[i], W=w_init_var, nonlinearity=lasagne.nonlinearities.softplus, b=b_init_var, name='ENC_A_var%i'%i)
        l_z[i] = SampleLayer(mu=lenc_z_mu[i], var=lenc_z_var[i], eq_samples=1, iw_samples=1)


    #DECODER MODEL
    ldec_mu_in = lasagne.layers.InputLayer((None,latent_sizes[-1]))
    ldec_var_in = lasagne.layers.InputLayer((None,latent_sizes[-1]))
    ldec_z = DecoderSampleLayer(l_z[-1],mu=ldec_mu_in,var=ldec_var_in)
    ldec_h = mlp(ldec_z, num_units=hidden_sizes[-1], W=w_init_mlp, name='ENC_Z%itoZ%i_DENSE'%(i,i-1), nonlinearity=nonlin_enc, num_mlp_layers=num_mlp_layers)

    for i in range(0,num_layers-1)[::-1]:
        ldec_z_mu[i] = denselayer(ldec_h, num_units=latent_sizes[i], W=w_init_mu, nonlinearity=lasagne.nonlinearities.identity, name='ENC_Z_MU%i'%i)
        ldec_z_var[i] = denselayer(ldec_h, num_units=latent_sizes[i], W=w_init_var, nonlinearity=lasagne.nonlinearities.softplus, b=b_init_var, name='ENC_Z_LOG_VAR%i'%i)
        ldec_z = DecoderSampleLayer(l_z[i],mu=ldec_z_mu[i],var=ldec_z_var[i])
        ldec_h = mlp(ldec_z, num_units=hidden_sizes[i], W=w_init_mlp, name='ENC_Z%itoZ%i_DENSE'%(i,i-1), nonlinearity=nonlin_enc, num_mlp_layers=num_mlp_layers)


    denselayerout = batchnormlayer if batch_norm_output else lasagne.layers.DenseLayer
    l_dec_x_mu = denselayerout(ldec_h, num_units=num_features, W=w_init_mu, nonlinearity=outputnonlin, name='DEC_DENSE_MU')
    l_dec_x_var = denselayerout(ldec_h, num_units=num_features, W=w_init_var, b=b_init_var, nonlinearity=lasagne.nonlinearities.softplus, name='DEC_DENSE_var')
    #note that the var layer is not used for anything if the density is set to bernoulli

elif modeltype == 'ladderVAE':
    reversed_z = True
    l_in = lasagne.layers.InputLayer((None, num_features))
    lenc_zl_mu = [[] for i in range(num_layers)]
    lenc_zl_var = [[] for i in range(num_layers)]
    l_z = [[] for i in range(num_layers)]
    lenc_z_mu = [[] for i in range(num_layers)]
    lenc_z_var = [[] for i in range(num_layers)]
    lenc_zt_mu = [[] for i in range(num_layers)]
    lenc_zt_var = [[] for i in range(num_layers)]
    lenc_zl_mu = [[] for i in range(num_layers)]
    lenc_zl_var = [[] for i in range(num_layers)]
    lenc_zc_mu = [[] for i in range(num_layers)]
    lenc_zc_var = [[] for i in range(num_layers)]
    ldec_z_mu = [[] for i in range(num_layers)]
    ldec_z_var = [[] for i in range(num_layers)]
    l_enc_a = [[] for i in range(num_layers)]

    l_enc_a[0]= mlp(l_in, num_units=hidden_sizes[0], W=w_init_mlp, name='ENC_A_DENSE%i'%0, nonlinearity=nonlin_enc, num_mlp_layers=num_mlp_layers)
    lenc_zl_mu[0] = denselayer(l_enc_a[0], num_units=latent_sizes[0], nonlinearity=lasagne.nonlinearities.identity, name='ENC_A_MU%i'%0)
    lenc_zl_var[0] = denselayer(l_enc_a[0], num_units=latent_sizes[0], nonlinearity=lasagne.nonlinearities.softplus, b=b_init_var, name='ENC_A_var%i'%0)


    for i in range(1,num_layers):
        if only_mu_up:
            l_enc_a[i]= mlp(lenc_zl_mu[i-1], num_units=hidden_sizes[i], W=w_init_mlp, name='ENC_A_DENSE%i'%i, nonlinearity=nonlin_enc, num_mlp_layers=num_mlp_layers)
        else:
            l_enc_a[i]= mlp(l_enc_a[i-1], num_units=hidden_sizes[i], W=w_init_mlp, name='ENC_A_DENSE%i'%i, nonlinearity=nonlin_enc, num_mlp_layers=num_mlp_layers)
        lenc_zl_mu[i] = denselayer(l_enc_a[i], num_units=latent_sizes[i], nonlinearity=lasagne.nonlinearities.identity, name='ENC_A_MU%i'%i)
        lenc_zl_var[i] = denselayer(l_enc_a[i], num_units=latent_sizes[i], nonlinearity=lasagne.nonlinearities.softplus, b=b_init_var, name='ENC_A_var%i'%i)

    lenc_z_mu[-1] = lenc_zl_mu[-1]
    lenc_z_var[-1] = lenc_zl_var[-1]
    l_z[-1] = SampleLayer(mu=lenc_z_mu[-1], var=lenc_z_var[-1], eq_samples=sym_eq_samples, iw_samples=sym_iw_samples)

    ldec_mu_in = lasagne.layers.InputLayer((None,latent_sizes[-1]))
    ldec_var_in = lasagne.layers.InputLayer((None,latent_sizes[-1]))
    ldec_z = DecoderSampleLayer(l_z[-1],mu=ldec_mu_in,var=ldec_var_in)
    #stupid naming og ldec_z and DecoderSampleLayer since they are actually part of the encoder in the ladder-VAE model
    #L-1 to 0 layer
    for i in range(0,num_layers-1)[::-1]:
        lenc_ztoz = mlp(ldec_z, num_units=hidden_sizes[i+1], W=w_init_mlp, name='ENC_Z%itoZ%i_DENSE'%(i+1,i), nonlinearity=nonlin_enc, num_mlp_layers=num_mlp_layers)
        lenc_zt_mu[i] = denselayer(lenc_ztoz, num_units=latent_sizes[i], nonlinearity=lasagne.nonlinearities.identity, name='ENC_Z_MU%i'%i)
        lenc_zt_var[i] = denselayer(lenc_ztoz, num_units=latent_sizes[i], nonlinearity=lasagne.nonlinearities.softplus, b=b_init_var, name='ENC_Z_LOG_VAR%i'%i)
        mlpout = LadderMergeLayer(mu_t=lenc_zt_mu[i], var_t=lenc_zt_var[i], mu_l=lenc_zl_mu[i], var_l=lenc_zl_var[i], eq_samples=sym_eq_samples, iw_samples=sym_iw_samples)
        lenc_z_mu[i], lenc_z_var[i] = ListIndexLayer(mlpout,index=0),ListIndexLayer(mlpout,index=1)
        l_z[i] = SampleLayer(mu=lenc_z_mu[i], var=lenc_z_var[i], eq_samples=1, iw_samples=1)
        #decoder model (also for sampling)
        if ladder_share_params:
            ldec_z_mu[i] = lenc_zt_mu[i]
            ldec_z_var[i] = lenc_zt_var[i]
        else:
            print "No sharing of parameters between ladder encoder and decoder"
            ldec_ztoz = mlp(ldec_z, num_units=hidden_sizes[i], W=w_init_mlp, name='DEC_Z%itoZ%i_DENSE'%(i+1,i), nonlinearity=nonlin_enc, num_mlp_layers=num_mlp_layers)
            ldec_z_mu[i] = denselayer(ldec_ztoz, num_units=latent_sizes[i], nonlinearity=lasagne.nonlinearities.identity, name='DEC_Z_MU%i'%i)
            ldec_z_var[i] = denselayer(ldec_ztoz, num_units=latent_sizes[i], nonlinearity=lasagne.nonlinearities.softplus, name='DEC_Z_MU%i'%i, b=b_init_var)

        ldec_z = DecoderSampleLayer(l_z[i],mu=ldec_z_mu[i],var=ldec_z_var[i])

    ldec_h = mlp(ldec_z, num_units=hidden_sizes[0], W=w_init_mlp, name='DEC_DENSE', nonlinearity=nonlin_dec, num_mlp_layers=num_mlp_layers)
    denselayerout = batchnormlayer if batch_norm_output else lasagne.layers.DenseLayer
    l_dec_x_mu = denselayerout(ldec_h, num_units=num_features, W=w_init_mu, nonlinearity=outputnonlin, name='DEC_DENSE_MU')
    l_dec_x_var = denselayerout(ldec_h, num_units=num_features, W=w_init_var, b=b_init_var, nonlinearity=lasagne.nonlinearities.softplus, name='DEC_DENSE_var')
else:
    raise ValueError()

# get output needed for evaluating model with noise if present
train_layers = lasagne.layers.get_output(l_z + lenc_z_mu + lenc_z_var + ldec_z_mu[:-1] + ldec_z_var[:-1] + [l_dec_x_mu, l_dec_x_var], {l_in:sym_x, ldec_mu_in:sym_mu, ldec_var_in:sym_var}, deterministic=False)
z_train = train_layers[:num_layers*1]
z_mu_q_train = train_layers[1*num_layers:2*num_layers]
z_var_q_train = train_layers[2*num_layers:3*num_layers]
z_mu_p_train = train_layers[3*num_layers:4*num_layers-1]
z_var_p_train = train_layers[4*num_layers-1:5*num_layers-2]
x_mu_train = train_layers[5*num_layers-2]
x_var_train = train_layers[5*num_layers-1]

test_layers = lasagne.layers.get_output(l_z + lenc_z_mu + lenc_z_var + ldec_z_mu[:-1] + ldec_z_var[:-1] + [l_dec_x_mu, l_dec_x_var], {l_in:sym_x, ldec_mu_in:sym_mu, ldec_var_in:sym_var}, deterministic=True)
z_test = test_layers[:num_layers*1]
z_mu_q_test = test_layers[1*num_layers:2*num_layers]
z_var_q_test = test_layers[2*num_layers:3*num_layers]
z_mu_p_test = test_layers[3*num_layers:4*num_layers-1]
z_var_p_test = test_layers[4*num_layers-1:5*num_layers-2]
x_mu_test = test_layers[5*num_layers-2]
x_var_test = test_layers[5*num_layers-1]

# TRAINING LOWER BOUND
#plotPCA('outfolder','pcatest',test_x,target=test_t)
def log_normal2_eps(x, mean, var, eps=1e-6):
    return -0.5 * math.log(2*math.pi) - var/2 - (x - mean)**2 / (2 * T.exp(var)+eps)


def latent_gaussian_x_visible(z, z_mu_q, z_var_q, z_mu_p, z_var_p, x_mu, x_var, x, eq_samples, iw_samples, latent_sizes, num_features,reverse_z=False, temp=1.0):
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
        z_var_q = [z_.reshape((-1, eq_samples, iw_samples, ls)) for z_,ls in zip(z_var_q[:-1],latent_sizes[:-1])] + [z_var_q[-1].dimshuffle((0,'x','x',1))]
    else:
        #for normal VAE where x->z1->z2->z3
        z = [z_.reshape((-1, eq_samples, iw_samples, ls)) for z_,ls in zip(z,latent_sizes)]
        z_mu_q = [z_mu_q[0].dimshuffle((0,'x','x',1))] + [z_.reshape((-1, eq_samples, iw_samples, ls)) for z_,ls in zip(z_mu_q[1:],latent_sizes[1:])]
        z_var_q =  [z_var_q[0].dimshuffle((0,'x','x',1))] + [z_.reshape((-1, eq_samples, iw_samples, ls)) for z_,ls in zip(z_var_q[1:],latent_sizes[1:])]

    z_mu_p = [z_.reshape((-1, eq_samples, iw_samples, ls)) for z_,ls in zip(z_mu_p,latent_sizes[:-1])]
    z_var_p = [z_.reshape((-1, eq_samples, iw_samples, ls)) for z_,ls in zip(z_var_p,latent_sizes[:-1])]

    x_mu = x_mu.reshape((-1, eq_samples, iw_samples,  num_features))
    x = x.dimshuffle((0,'x','x',1))

    if outputdensity == 'bernoulli':
        log_px = -T.nnet.binary_crossentropy(T.clip(x_mu,lv_eps_out,1-lv_eps_out), x)
    elif outputdensity == 'gaussian':
        x_var = x_var.reshape((-1, eq_samples, iw_samples,  num_features))
        log_px = log_normal2(x, x_mu, T.log(x_var+lv_eps_out))
    elif outputdensity == 'multinomial':
        log_px = -T.nnet.categorical_crossentropy(T.clip(x_mu,lv_eps_out,1-lv_eps_out), x)
    if outputdensity == 'bernoulli_logloss':
        log_px = x*T.log(T.clip(x_mu,lv_eps_out,1-lv_eps_out))

    log_pz =  [log_normal2(z_, mu_, T.log(var_+lv_eps_z)) for z_, mu_, var_ in zip(z[:-1],z_mu_p,z_var_p)] + [log_stdnormal(z[-1])]
    log_qz = [log_normal2(z_, mu_, T.log(var_+lv_eps_z)) for z_, mu_, var_ in zip(z, z_mu_q,z_var_q)]

    #all log_*** should have dimension (batch_size, nsamples, ivae_samples)
    a = log_px.sum(axis=3) + temp*(sum([p.sum(axis=3) for p in log_pz]) - sum([p.sum(axis=3) for p in log_qz]))
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

lower_bound_train, log_px_train, log_pz_train, log_qz_train = latent_gaussian_x_visible(z_train, z_mu_q_train, z_var_q_train, z_mu_p_train, z_var_p_train, x_mu_train, x_var_train, sym_x, eq_samples=sym_eq_samples, iw_samples=sym_iw_samples, num_features=num_features, latent_sizes=latent_sizes, reverse_z=reversed_z, temp=sym_temp)
lower_bound_test, log_px_test, log_pz_test, log_qz_test = latent_gaussian_x_visible(z_test, z_mu_q_test, z_var_q_test, z_mu_p_test, z_var_p_test, x_mu_test, x_var_test, sym_x, eq_samples=sym_eq_samples, iw_samples=sym_iw_samples, num_features=num_features, latent_sizes=latent_sizes, reverse_z=reversed_z)


print "lower_bound_train", lower_bound_train.eval({sym_x:X, sym_iw_samples: np.int32(iw_samples), sym_eq_samples:np.int32(eq_samples), sym_temp:np.float32(1.0)})

line = ''
outputlayers = [l_dec_x_mu] + ldec_z_mu[:-1] + ldec_z_var[:-1]
if outputdensity == 'gaussian':
    outputlayers += [l_dec_x_var]

params = lasagne.layers.get_all_params(outputlayers, trainable=True)
for p in params:
    print p, p.get_value().shape
    line += "%s %s\n" % (p, str(p.get_value().shape))

with open(trainlogfile,'w') as f:
    f.write("Trainlog\n")

with open(logfile,'a') as f:
    f.write(line)

cost = -lower_bound_train
if L2 is not 0:
    print "using L2 reg of %0.2e"%L2
    cost += sum(T.mean(p**2) for p in params)*L2

### note the minus because we want to push up the lowerbound
grads = T.grad(cost, params)
clip_grad = 0.9 # changed here from $
max_norm = 4 # changed here from 5
mgrads = lasagne.updates.total_norm_constraint(grads,max_norm=max_norm)
cgrads = [T.clip(g,-clip_grad, clip_grad) for g in mgrads]

updates = lasagne.updates.adam(cgrads, params,beta1=0.9, beta2=0.999, epsilon=1e-4, learning_rate=sym_lr)

if drawsamples:
    xsample = lasagne.layers.get_output(l_dec_x_mu,{ldec_mu_in:sym_mu, ldec_var_in:sym_var, l_in:sym_x}, deterministic=True, drawdecsample=True)
    f_sample = theano.function([sym_x, sym_mu, sym_var,sym_eq_samples, sym_iw_samples],xsample,on_unused_input='ignore')


train_model = theano.function([sym_x, sym_lr, sym_eq_samples, sym_iw_samples, sym_temp],
                              [lower_bound_train] +
                              [log_px_train.sum(axis=3).mean()] +
                              [p.sum(axis=3).mean() for p in log_pz_train] +
                              [p.sum(axis=3).mean() for p in log_qz_train] +
                              [T.mean(p1-p2,axis=(0,1)) for p1,p2 in zip(log_qz_train,log_pz_train)] +
                              z_mu_p_train +
                              z_var_p_train +
                              z_mu_q_train +
                              z_var_q_train +
                              z_train,
                              updates=updates)


test_model = theano.function([sym_x,sym_eq_samples, sym_iw_samples],
                              [lower_bound_test] +
                              [log_px_test.sum(axis=3).mean()] +
                              [p.sum(axis=3).mean() for p in log_pz_test] +
                              [p.sum(axis=3).mean() for p in log_qz_test] +
                              [T.mean(p1-p2,axis=(0,1)) for p1,p2 in zip(log_qz_test,log_pz_test)] +
                              z_mu_p_test +
                              z_var_p_test +
                              z_mu_q_test +
                              z_var_q_test +
                              z_test +
                              [x_mu_test])

test_model5000 = theano.function([sym_x,sym_eq_samples, sym_iw_samples],
                              [lower_bound_test] +
                              [log_px_test.sum(axis=3).mean()] +
                              [p.sum(axis=3).mean() for p in log_pz_test] +
                              [p.sum(axis=3).mean() for p in log_qz_test] +
                              [T.mean(p1-p2,axis=(0,1)) for p1,p2 in zip(log_qz_test,log_pz_test)])


if batch_norm:
    try:
        collect_x = process_data(collect_x) #if defined use the sh_x_collect for bn
    except:
        collect_x = process_data(train_x)  #else just use the full training data
    collect_out = lasagne.layers.get_output(outputlayers,{l_in:sym_x, ldec_mu_in:sym_mu, ldec_var_in:sym_var}, deterministic=True, collect=True)
    f_collect = theano.function([sym_x, sym_eq_samples, sym_iw_samples],
                                collect_out)


n_train_batches = train_x.shape[0] / batch_size
#n_valid_batches = valid_x.shape[0] / batch_size_val
n_test_batches = test_x.shape[0] / batch_size_test


def train_epoch(x,lr,eq_samples,iw_samples,epoch,temp):
    costs, log_px = [],[],
    log_pz = []
    log_qz = []
    KL_qp = None
    mu_p = []
    var_p = []
    mu_q = []
    var_q = []
    z_sample = []

    for j in range(num_layers-1):
        mu_p += [np.zeros((train_x.shape[0],eq_samples,iw_samples,latent_sizes[j]))]
        var_p += [np.zeros((train_x.shape[0],eq_samples,iw_samples,latent_sizes[j]))]

    for j in range(num_layers):
        if reversed_z:
            mu_q += [np.zeros((train_x.shape[0],1,1,latent_sizes[j]))] if j == num_layers-1 else [np.zeros((train_x.shape[0],eq_samples,iw_samples,latent_sizes[j]))]
            var_q += [np.zeros((train_x.shape[0],1,1,latent_sizes[j]))] if j == num_layers-1 else [np.zeros((train_x.shape[0],eq_samples,iw_samples,latent_sizes[j]))]
        else:
            mu_q += [np.zeros((train_x.shape[0],1,1,latent_sizes[j]))] if j == 0 else [np.zeros((train_x.shape[0],eq_samples,iw_samples,latent_sizes[j]))]
            var_q += [np.zeros((train_x.shape[0],1,1,latent_sizes[j]))] if j == 0 else [np.zeros((train_x.shape[0],eq_samples,iw_samples,latent_sizes[j]))]

    for j in range(num_layers):
        z_sample += [np.zeros((train_x.shape[0],eq_samples,iw_samples,latent_sizes[j]))]

    for i in range(n_train_batches):
        x_batch = x[i*batch_size:(i+1)*batch_size]
        #if epoch == 1:
        #    lr = lr*1.0/float(n_train_batches-i)
        out = train_model(x_batch,lr,eq_samples,iw_samples,temp)

        costs += [out[0]]
        log_px += [out[1]]
        log_pz += [out[2:2+num_layers]]
        log_qz += [out[2+1*num_layers:2+2*num_layers]]
        verbose_print([str(i)] + map(lambda s: "%0.2f"%s,[out[0]]+ [out[1]] + out[2:2+num_layers] + out[2+1*num_layers:2+2*num_layers]))
        if KL_qp == None:
            KL_qp = out[2+2*num_layers:2+3*num_layers]
        else:
            KL_qp = [old+new for old,new in zip(KL_qp, out[2+2*num_layers:2+3*num_layers])]

        if epoch in eval_epochs:
            mu_p_batch = out[2+3*num_layers:1+4*num_layers]
            var_p_batch = out[1+4*num_layers:0+5*num_layers]
            for j,mu in enumerate(mu_p_batch):
                mu_p[j][i*batch_size:(i+1)*batch_size] = mu.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))

            for j,var in enumerate(var_p_batch):
                var_p[j][i*batch_size:(i+1)*batch_size] = var.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))

            mu_q_batch = out[0+5*num_layers:0+6*num_layers]
            var_q_batch = out[0+6*num_layers:0+7*num_layers]
            for j,mu in enumerate(mu_q_batch):
                if reversed_z:
                    mu_q[j][i*batch_size:(i+1)*batch_size] = mu.reshape((-1,1,1,latent_sizes[j])) if j == num_layers-1 else mu.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))
                else:
                    mu_q[j][i*batch_size:(i+1)*batch_size] = mu.reshape((-1,1,1,latent_sizes[j])) if j == 0 else mu.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))


            for j,var in enumerate(var_q_batch):
                if reversed_z:
                    var_q[j][i*batch_size:(i+1)*batch_size] = var.reshape((-1,1,1,latent_sizes[j])) if j == num_layers-1 else var.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))
                else:
                    var_q[j][i*batch_size:(i+1)*batch_size] = var.reshape((-1,1,1,latent_sizes[j])) if j == 0 else var.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))

            z_batch = out[0+7*num_layers:0+8*num_layers]
            for j,z in enumerate(z_batch):
                z_sample[j][i*batch_size:(i+1)*batch_size] = z.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))


    if epoch in eval_epochs:
        for j in range(num_layers-1):
            boxplot(res_out,'epoch%i_train_boxplot_mu_p_layer%i'%(epoch,j),mu_p[j])
            boxplot(res_out,'epoch%i_train_boxplot_var_p_layer%i'%(epoch,j),var_p[j])

        for j in range(num_layers):
            boxplot(res_out,'epoch%i_train_boxplot_mu_q_layer%i'%(epoch,j),mu_q[j])
            boxplot(res_out,'epoch%i_train_boxplot_var_q_layer%i'%(epoch,j),var_q[j])

        for j in range(num_layers):
            boxplot(res_out,'epoch%i_train_boxplot_zsample_layer%i'%(epoch,j),z_sample[j])



    return np.mean(costs), np.mean(log_px,axis=0), \
           np.mean(log_pz,axis=0), np.mean(log_qz,axis=0), \
           [KL/float(n_train_batches) for KL in KL_qp]


def test_epoch(x,eq_samples,iw_samples):
    if batch_norm:
        _ = f_collect(collect_x,1,1) #collect BN stats on train
    costs, log_px = [],[],
    log_pz = []
    log_qz = []
    KL_qp = None
    mu_p = []
    var_p = []
    mu_q = []
    var_q = []
    z_sample = []
    if iw_samples == 1 and eq_samples == 1:
        for j in range(num_layers-1):
            mu_p += [np.zeros((test_x.shape[0],eq_samples,iw_samples,latent_sizes[j]))]
            var_p += [np.zeros((test_x.shape[0],eq_samples,iw_samples,latent_sizes[j]))]

        for j in range(num_layers):
            mu_q += [np.zeros((test_x.shape[0],1,1,latent_sizes[j]))] if j == 0 else [np.zeros((test_x.shape[0],eq_samples,iw_samples,latent_sizes[j]))]
            var_q += [np.zeros((test_x.shape[0],1,1,latent_sizes[j]))] if j == 0 else [np.zeros((test_x.shape[0],eq_samples,iw_samples,latent_sizes[j]))]

        for j in range(num_layers):
            z_sample += [np.zeros((test_x.shape[0],eq_samples,iw_samples,latent_sizes[j]))]

    if iw_samples == 1 and eq_samples == 1:
        model = test_model
    else:
        model = test_model5000

    for i in range(n_test_batches):
        x_batch = x[i*batch_size_test:(i+1)*batch_size_test]
        out = model(x_batch,eq_samples,iw_samples)

        costs += [out[0]]
        log_px += [out[1]]
        log_pz += [out[2:2+num_layers]]
        log_qz += [out[2+1*num_layers:2+2*num_layers]]
        if KL_qp == None:
            KL_qp = out[2+2*num_layers:2+3*num_layers]
        else:
            KL_qp = [old+new for old,new in zip(KL_qp, out[2+2*num_layers:2+3*num_layers])]

        if iw_samples == 1 and eq_samples == 1: #dont want to do this for eq5000 since it is a lot of samples
            mu_p_batch = out[2+3*num_layers:1+4*num_layers]
            var_p_batch = out[1+4*num_layers:0+5*num_layers]
            for j,mu in enumerate(mu_p_batch):
                mu_p[j][i*batch_size_test:(i+1)*batch_size_test] = mu.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))

            for j,var in enumerate(var_p_batch):
                var_p[j][i*batch_size_test:(i+1)*batch_size_test] = var.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))

            mu_q_batch = out[0+5*num_layers:0+6*num_layers]
            var_q_batch = out[0+6*num_layers:0+7*num_layers]
            for j,mu in enumerate(mu_q_batch):
                mu_q[j][i*batch_size_test:(i+1)*batch_size_test] = mu.reshape((-1,1,1,latent_sizes[j])) if j == 0 else mu.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))

            for j,var in enumerate(var_q_batch):
                var_q[j][i*batch_size_test:(i+1)*batch_size_test] = var.reshape((-1,1,1,latent_sizes[j])) if j == 0 else var.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))

            z_batch = out[0+7*num_layers:0+8*num_layers]
            for j,z in enumerate(z_batch):
                z_sample[j][i*batch_size_test:(i+1)*batch_size_test] = z.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))

    if iw_samples == 1 and eq_samples == 1:
        for j in range(num_layers-1):
            boxplot(res_out,'epoch%i_test_boxplot_mu_p_layer%i'%(epoch,j),mu_p[j])
            boxplot(res_out,'epoch%i_test_boxplot_var_p_layer%i'%(epoch,j),var_p[j])

        for j in range(num_layers):
            boxplot(res_out,'epoch%i_test_boxplot_mu_q_layer%i'%(epoch,j),mu_q[j])
            boxplot(res_out,'epoch%i_test_boxplot_var_q_layer%i'%(epoch,j),var_q[j])

        for j in range(num_layers):
            boxplot(res_out,'epoch%i_test_boxplot_zsample_layer%i'%(epoch,j),z_sample[j])

            if pcaplot:
                plotPCA(res_out,'epoch%i_test_pca_zsample_layer%i'%(epoch,j),z_sample[j].squeeze(),target=test_t)



    return np.mean(costs), np.mean(log_px,axis=0), \
           np.mean(log_pz,axis=0), np.mean(log_qz,axis=0), \
           [KL/float(n_test_batches) for KL in KL_qp]

total_time_start = time.time()
train_res = init_res(num_layers)
test1_res = init_res(num_layers)
test5000_res = init_res(num_layers)
temp_step = (1-temp_start) / float(temp_epochs)
temp = temp_start
print "Training"

for epoch in range(1,num_epochs+1):
    start = time.time()
    #if epoch > 2000:
    #    lr = lr*0.9995

    np.random.shuffle(train_x)

    if ramp_n_samples:
        eq_samples_cur = min(eq_samples,epoch)
        iw_samples_cur = min(iw_samples,epoch)
    else:
        eq_samples_cur, iw_samples_cur = eq_samples, iw_samples

    train_out = train_epoch(process_data(train_x),lr,eq_samples_cur, iw_samples_cur, epoch, temp)
    costs_train_tmp, log_px_train_tmp, log_pz_cur_train, log_qz_cur_train, KL_tmp = train_out
    t = time.time() - start
    line = "*Epoch=%i\tTime=%0.2f\tLR=%0.5f\tE_qsamples=%i\tIVAEsamples=%i\ttemp: %0.2f\t" %(epoch, t, lr, eq_samples, iw_samples,temp) + \
        "TRAIN:\tCost=%0.5f\tlogp(x|z1)=%0.5f\t"%(costs_train_tmp, log_px_train_tmp) + \
        "log p(z): " + "|".join(map(lambda s: "%0.3f"%s,log_pz_cur_train)) + "\t"  + \
        "log q(z): " + "|".join(map(lambda s: "%0.3f"%s,log_qz_cur_train))

    print line
    with open(trainlogfile,'a') as f:
        f.write(line + "\n")

    temp = min(temp + temp_step,1.0)

    if np.isnan(train_out[0]):
        break

    if epoch in eval_epochs:
        t = time.time() - start #stop time so we only measure train time
        print "calculating L1, L5000"

        costs_train_tmp, log_px_train_tmp, log_pz_cur_train, log_qz_cur_train, KL_tmp = train_out
        train_res = add_res(train_out,epoch,train_res)

        test1_out = test_epoch(test_x,1,1)
        costs_test1_tmp, log_px_test1_tmp, log_pz_cur_test1, log_qz_cur_test1, KL_tmp = test1_out
        test1_res = add_res(test1_out,epoch,test1_res)

        test5000_out = test_epoch(test_x,1,5000)
        costs_test5000_tmp, log_px_test5000_tmp, log_pz_cur_test5000, log_qz_cur_test5000, KL_tmp = test5000_out
        test5000_res = add_res(test5000_out,epoch,test5000_res)

        with open(res_out + '/res.cpkl','w') as f:
            cPickle.dump([train_res,test1_res,test5000_res],f,protocol=cPickle.HIGHEST_PROTOCOL)

        if drawsamples:
            print "drawing samples"
            mu_sample = np.zeros((10**2,latent_sizes[-1])).astype('float32')
            var_sample = np.ones((10**2,latent_sizes[-1])).astype('float32')
            dummyX = np.zeros((10**2,num_features)).astype('float32')
            samples =  f_sample(dummyX,  mu_sample, var_sample,1,1) #get rid of sym_x maybe?
            plotsamples('samples_prior%i'%epoch,res_out,samples.reshape([-1]+imgshp))
            samples = test_model(test_x[:10**2],1,1)[-1]
            plotsamples('samples_conditioned_%i'%epoch,res_out,samples.reshape([-1]+imgshp))
            #plotsamples('samples_conditioned_true_%i'%epoch,res_out,sh_x_test.get_value()[:10**2].reshape([-1]+imgshp))

        #dump model params
        all_params=lasagne.layers.get_all_param_values(outputlayers)
        f = gzip.open(model_out + 'epoch%i'%(epoch), 'wb')
        cPickle.dump(all_params, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

        # LOGGING, SAVING MODEL and PLOTTING

        line = "*Epoch=%i\tTime=%0.2f\tLR=%0.5f\tE_qsamples=%i\tIVAEsamples=%i\t" %(epoch, t, lr, eq_samples, iw_samples) + \
            "TRAIN:\tCost=%0.5f\tlogp(x|z1)=%0.5f\t"%(train_res['cost'][-1], train_res['log_px'][-1]) + \
            "log p(z): " + "|".join(map(lambda s: "%0.3f"%s,log_pz_cur_train)) + "\t"  + \
            "log q(z): " + "|".join(map(lambda s: "%0.3f"%s,log_qz_cur_train)) + "\t"  + \
            "TEST-1:\tCost=%0.5f\tlogp(x|z1)=%0.5f\t"%(test1_res['cost'][-1], test1_res['log_px'][-1]) + \
            "log p(z): " + "|".join(map(lambda s: "%0.3f"%s,log_pz_cur_test1)) + "\t"  + \
            "log q(z): " + "|".join(map(lambda s: "%0.3f"%s,log_qz_cur_test1)) + "\t"  + \
            "TEST-5000:\tCost=%0.5f\tlogp(x|z1)=%0.5f\t"%(test5000_res['cost'][-1], test5000_res['log_px'][-1]) + \
            "log p(z): " + "|".join(map(lambda s: "%0.3f"%s,log_pz_cur_test5000)) + "\t"  + \
            "log q(z): " + "|".join(map(lambda s: "%0.3f"%s,log_qz_cur_test5000)) + "\t" + \
            "%0.5f\t%0.5f\t%0.5f" %(train_res['cost'][-1],test1_res['cost'][-1],test5000_res['cost'][-1])


        print line

        with open(logfile,'a') as f:
            f.write(line + "\n")

        plotLLs('Train_LLs',res_out,train_res['epoch'],train_res['cost'],train_res['log_px'],train_res['log_pz'],train_res['log_qz'])
        plotLLs('Test1_LLs',res_out,test1_res['epoch'],test1_res['cost'],test1_res['log_px'],test1_res['log_pz'],test1_res['log_qz'])
        plotLLs('Test5000_LLs',res_out,test5000_res['epoch'],test5000_res['cost'],test5000_res['log_px'],test5000_res['log_pz'],test5000_res['log_qz'])
        for i,KL in enumerate(train_res['KL_qp']):
            plotKLs('Train_KL_z%i'%i,res_out,train_res['epoch'],KL)

        for i,KL in enumerate(test1_res['KL_qp']):
            plotKLs('Test1_KL_z%i'%i,res_out,test1_res['epoch'],KL)

        for i,KL in enumerate(test5000_res['KL_qp']):
            plotKLs('Test5000_KL_z%i'%i,res_out,test5000_res['epoch'],KL)

        plt.close("all")
