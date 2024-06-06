"""ALI-related bricks."""
from theano import tensor
from blocks.bricks.base import Brick, application, lazy
from blocks.bricks.conv import ConvolutionalSequence
from blocks.bricks.interfaces import Initializable, Random
from blocks.select import Selector


class ALI(Initializable, Random):
    """Adversarial learned inference brick.

    Parameters
    ----------
    encoder : :class:`blocks.bricks.Brick`
        Encoder network.
    decoder : :class:`blocks.bricks.Brick`
        Decoder network.
    discriminator : :class:`blocks.bricks.Brick`
        Discriminator network taking :math:`x` and :math:`z` as input.

    """
    def __init__(self, encoder, decoder, discriminator, **kwargs):
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

        super(ALI, self).__init__(**kwargs)
        self.children.extend([self.encoder, self.decoder, self.discriminator])

    @property
    def discriminator_parameters(self):
        return list(
            Selector([self.discriminator]).get_parameters().values())

    @property
    def generator_parameters(self):
        return list(
            Selector([self.encoder, self.decoder]).get_parameters().values())

    @application(inputs=['x', 'z_hat', 'x_tilde', 'z'],
                 outputs=['data_preds', 'sample_preds'])
    def get_predictions(self, x, z_hat, x_tilde, z, application_call):
        # NOTE: the unbroadcasts act as a workaround for a weird broadcasting
        # bug when applying dropout
        input_x = tensor.unbroadcast(
            tensor.concatenate([x, x_tilde], axis=0), *range(x.ndim))
        input_z = tensor.unbroadcast(
            tensor.concatenate([z_hat, z], axis=0), *range(x.ndim))
        data_sample_preds = self.discriminator.apply(input_x, input_z)
        data_preds = data_sample_preds[:x.shape[0]]
        sample_preds = data_sample_preds[x.shape[0]:]

        application_call.add_auxiliary_variable(
            tensor.nnet.sigmoid(data_preds).mean(), name='data_accuracy')
        application_call.add_auxiliary_variable(
            (1 - tensor.nnet.sigmoid(sample_preds)).mean(),
            name='sample_accuracy')

        return data_preds, sample_preds

    @application(inputs=['x', 'z'],
                 outputs=['discriminator_loss', 'generator_loss'])
    def compute_losses(self, x, z, application_call):
        z_hat = self.encoder.apply(x)
        x_tilde = self.decoder.apply(z)

        data_preds, sample_preds = self.get_predictions(x, z_hat, x_tilde, z)

        discriminator_loss = (tensor.nnet.softplus(-data_preds) +
                              tensor.nnet.softplus(sample_preds)).mean()
        generator_loss = (tensor.nnet.softplus(data_preds) +
                          tensor.nnet.softplus(-sample_preds)).mean()

        return discriminator_loss, generator_loss

    @application(inputs=['z'], outputs=['samples'])
    def sample(self, z):
        return self.decoder.apply(z)

    @application(inputs=['x'], outputs=['reconstructions'])
    def reconstruct(self, x):
        return self.decoder.apply(self.encoder.apply(x))


class COVConditional(Initializable, Random):
    """Change-of-variables conditional.

    Parameters
    ----------
    mapping : :class:`blocks.bricks.Brick`
        Network mapping the concatenation of the input and a source of
        noise to the output.
    noise_shape : tuple of int
        Shape of the input noise.

    """
    def __init__(self, mapping, noise_shape, **kwargs):
        self.mapping = mapping
        self.noise_shape = noise_shape

        super(COVConditional, self).__init__(**kwargs)
        self.children.extend([self.mapping])

    def get_dim(self, name):
        if isinstance(self.mapping, ConvolutionalSequence):
            dim = self.mapping.get_dim(name)
            if name == 'input_':
                return (dim[0] - self.noise_shape[0],) + dim[1:]
            else:
                return dim
        else:
            if name == 'output':
                return self.mapping.output_dim
            elif name == 'input_':
                return self.mapping.input_dim - self.noise_shape[0]
            else:
                return self.mapping.get_dim(name)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_, application_call):
        epsilon = self.theano_rng.normal(
            size=(input_.shape[0],) + self.noise_shape)
        output = self.mapping.apply(
            tensor.concatenate([input_, epsilon], axis=1))

        application_call.add_auxiliary_variable(output.mean(), name='avg')
        application_call.add_auxiliary_variable(output.std(), name='std')
        application_call.add_auxiliary_variable(output.min(), name='min')
        application_call.add_auxiliary_variable(output.max(), name='max')

        return output


class GaussianConditional(Initializable, Random):
    """Gaussian conditional.

    Parameters
    ----------
    mapping : :class:`blocks.bricks.Brick`
        Network predicting distribution parameters. It is expected to
        output a concatenation of :math:`\mu` and :math:`\log\sigma`.

    """
    def __init__(self, mapping, **kwargs):
        self.mapping = mapping

        super(GaussianConditional, self).__init__(**kwargs)
        self.children.extend([self.mapping])

    @property
    def _nlat(self):
        if isinstance(self.mapping, ConvolutionalSequence):
            return self.get_dim('output')[0]
        else:
            return self.get_dim('output')

    def get_dim(self, name):
        if isinstance(self.mapping, ConvolutionalSequence):
            dim = self.mapping.get_dim(name)
            if name == 'output':
                return (dim[0] // 2,) + dim[1:]
            else:
                return dim
        else:
            if name == 'output':
                return self.mapping.output_dim // 2
            elif name == 'input_':
                return self.mapping.input_dim
            else:
                return self.mapping.get_dim(name)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_, application_call):
        params = self.mapping.apply(input_)
        mu, log_sigma = params[:, :self._nlat], params[:, self._nlat:]
        sigma = tensor.exp(log_sigma)
        epsilon = self.theano_rng.normal(size=mu.shape)
        output = mu + sigma * epsilon

        application_call.add_auxiliary_variable(mu.mean(), name='mu_avg')
        application_call.add_auxiliary_variable(mu.std(), name='mu_std')
        application_call.add_auxiliary_variable(mu.min(), name='mu_min')
        application_call.add_auxiliary_variable(mu.max(), name='mu_max')

        application_call.add_auxiliary_variable(sigma.mean(), name='sigma_avg')
        application_call.add_auxiliary_variable(sigma.std(), name='sigma_std')
        application_call.add_auxiliary_variable(sigma.min(), name='sigma_min')
        application_call.add_auxiliary_variable(sigma.max(), name='sigma_max')

        return output


class DeterministicConditional(Initializable, Random):
    """Deterministic conditional.

    Parameters
    ----------
    mapping : :class:`blocks.bricks.Brick`
        Network producing the output of the conditional.

    """
    def __init__(self, mapping, **kwargs):
        self.mapping = mapping

        super(DeterministicConditional, self).__init__(**kwargs)
        self.children.extend([self.mapping])

    def get_dim(self, name):
        return self.mapping.get_dim(name)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_, application_call):
        output = self.mapping.apply(input_)

        application_call.add_auxiliary_variable(output.mean(), name='avg')
        application_call.add_auxiliary_variable(output.std(), name='std')
        application_call.add_auxiliary_variable(output.min(), name='min')
        application_call.add_auxiliary_variable(output.max(), name='max')

        return output


class XZJointDiscriminator(Initializable):
    """Three-way discriminator.

    Parameters
    ----------
    x_discriminator : :class:`blocks.bricks.Brick`
        Part of the discriminator taking :math:`x` as input. Its
        output will be concatenated with ``z_discriminator``'s output
        and fed to ``joint_discriminator``.
    z_discriminator : :class:`blocks.bricks.Brick`
        Part of the discriminator taking :math:`z` as input. Its
        output will be concatenated with ``x_discriminator``'s output
        and fed to ``joint_discriminator``.
    joint_discriminator : :class:`blocks.bricks.Brick`
        Part of the discriminator taking the concatenation of
        ``x_discriminator``'s and output ``z_discriminator``'s output
        as input and computing :math:`D(x, z)`.

    """
    def __init__(self, x_discriminator, z_discriminator, joint_discriminator,
                 **kwargs):
        self.x_discriminator = x_discriminator
        self.z_discriminator = z_discriminator
        self.joint_discriminator = joint_discriminator

        super(XZJointDiscriminator, self).__init__(**kwargs)
        self.children.extend([self.x_discriminator, self.z_discriminator,
                              self.joint_discriminator])

    @application(inputs=['x', 'z'], outputs=['output'])
    def apply(self, x, z):
        # NOTE: the unbroadcasts act as a workaround for a weird broadcasting
        # bug when applying dropout
        input_ = tensor.unbroadcast(
            tensor.concatenate(
                [self.x_discriminator.apply(x), self.z_discriminator.apply(z)],
                axis=1),
            *range(x.ndim))
        return self.joint_discriminator.apply(input_)


class GAN(Initializable, Random):
    """Generative adversarial networks.

    Parameters
    ----------
    decoder : :class:`blocks.bricks.Brick`
        Decoder network.
    discriminator : :class:`blocks.bricks.Brick`
        Discriminator network.

    """
    def __init__(self, decoder, discriminator, **kwargs):
        self.decoder = decoder
        self.discriminator = discriminator

        super(GAN, self).__init__(**kwargs)
        self.children.extend([self.decoder, self.discriminator])

    @property
    def discriminator_parameters(self):
        return list(
            Selector([self.discriminator]).get_parameters().values())

    @property
    def generator_parameters(self):
        return list(
            Selector([self.decoder]).get_parameters().values())

    @application(inputs=['z'], outputs=['x_tilde'])
    def sample_x_tilde(self, z, application_call):
        x_tilde = self.decoder.apply(z)

        application_call.add_auxiliary_variable(x_tilde.mean(), name='avg')
        application_call.add_auxiliary_variable(x_tilde.std(), name='std')

        return x_tilde

    @application(inputs=['x', 'x_tilde'],
                 outputs=['data_preds', 'sample_preds'])
    def get_predictions(self, x, x_tilde, application_call):
        # NOTE: the unbroadcasts act as a workaround for a weird broadcasting
        # bug when applying dropout
        data_sample_preds = self.discriminator.apply(
            tensor.unbroadcast(tensor.concatenate([x, x_tilde], axis=0),
                               *range(x.ndim)))
        data_preds = data_sample_preds[:x.shape[0]]
        sample_preds = data_sample_preds[x.shape[0]:]

        application_call.add_auxiliary_variable(
            tensor.nnet.sigmoid(data_preds).mean(), name='data_accuracy')
        application_call.add_auxiliary_variable(
            (1 - tensor.nnet.sigmoid(sample_preds)).mean(),
            name='sample_accuracy')

        return data_preds, sample_preds

    @application(inputs=['x'],
                 outputs=['discriminator_loss', 'generator_loss'])
    def compute_losses(self, x, z, application_call):
        x_tilde = self.sample_x_tilde(z)
        data_preds, sample_preds = self.get_predictions(x, x_tilde)

        discriminator_loss = (tensor.nnet.softplus(-data_preds) +
                              tensor.nnet.softplus(sample_preds)).mean()
        generator_loss = (tensor.nnet.softplus(data_preds) +
                          tensor.nnet.softplus(-sample_preds)).mean()

        return discriminator_loss, generator_loss

    @application(inputs=['z'], outputs=['samples'])
    def sample(self, z):
        return self.sample_x_tilde(z)


class ConvMaxout(Brick):
    """Convolutional version of the Maxout activation.

    Parameters
    ----------
    num_pieces : int
        Number of linear pieces.
    num_channels : int
        Number of input channels.
    image_size : (int, int), optional
        Input shape. Defaults to ``(None, None)``.

    """
    @lazy(allocation=['num_pieces', 'num_channels'])
    def __init__(self, num_pieces, num_channels, image_size=(None, None),
                 **kwargs):
        super(ConvMaxout, self).__init__(**kwargs)
        self.num_pieces = num_pieces
        self.num_channels = num_channels

    def get_dim(self, name):
        if name == 'input_':
            return (self.num_channels,) + self.image_size
        if name == 'output':
            return (self.num_filters,) + self.image_size
        return super(ConvMaxout, self).get_dim(name)

    @property
    def num_filters(self):
        return self.num_channels // self.num_pieces

    @property
    def num_output_channels(self):
        return self.num_filters

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        input_ = input_.dimshuffle(0, 2, 3, 1)
        new_shape = ([input_.shape[i] for i in range(input_.ndim - 1)] +
                     [self.num_filters, self.num_pieces])
        output = tensor.max(input_.reshape(new_shape, ndim=input_.ndim + 1),
                            axis=input_.ndim)
        return output.dimshuffle(0, 3, 1, 2)
