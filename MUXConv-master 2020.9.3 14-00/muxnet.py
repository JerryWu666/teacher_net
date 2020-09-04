"""
based on EfficientNet implementation from
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py
"""
import re
import math
import logging
from copy import deepcopy

from conv2d_helpers import select_conv2d

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.registry import register_model
from timm.models.helpers import load_pretrained
from timm.models.helpers import load_checkpoint
import os
####

#_logger = logging.getLogger(__name__)


#def load_state_dict(checkpoint_path, use_ema=False):
#    if checkpoint_path and os.path.isfile(checkpoint_path):
#        checkpoint = torch.load(checkpoint_path, map_location='cpu')
#        state_dict_key = 'state_dict'
#        if isinstance(checkpoint, dict):
#            if use_ema and 'state_dict_ema' in checkpoint:
#                state_dict_key = 'state_dict_ema'
#        if state_dict_key and state_dict_key in checkpoint:
#            new_state_dict = OrderedDict()
#            for k, v in checkpoint[state_dict_key].items():
#                # strip `module.` prefix
#                name = k[7:] if k.startswith('module') else k
#                new_state_dict[name] = v
#            state_dict = new_state_dict
#        else:
#            state_dict = checkpoint
#        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
#        return state_dict
#    else:
#        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
#        raise FileNotFoundError()


#def load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
#    state_dict = load_state_dict(checkpoint_path, use_ema)
#    model.load_state_dict(state_dict, strict=strict)


#def resume_checkpoint(model, checkpoint_path):
#    other_state = {}
#    resume_epoch = None
#    if os.path.isfile(checkpoint_path):
#        checkpoint = torch.load(checkpoint_path, map_location='cpu')
#        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
#            new_state_dict = OrderedDict()
#            for k, v in checkpoint['state_dict'].items():
#                name = k[7:] if k.startswith('module') else k
#                new_state_dict[name] = v
#            model.load_state_dict(new_state_dict)
#            if 'optimizer' in checkpoint:
#                other_state['optimizer'] = checkpoint['optimizer']
#            if 'amp' in checkpoint:
#                other_state['amp'] = checkpoint['amp']
#            if 'epoch' in checkpoint:
#                resume_epoch = checkpoint['epoch']
#                if 'version' in checkpoint and checkpoint['version'] > 1:
#                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save
#            _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
#        else:
#            model.load_state_dict(checkpoint)
#            _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
#        return other_state, resume_epoch
#    else:
#        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
#        raise FileNotFoundError()


#def load_pretrained(model, cfg=None, num_classes=1000, in_chans=3, filter_fn=None, strict=True):
#    if cfg is None:
#        cfg = getattr(model, 'default_cfg')
#    if cfg is None or 'url' not in cfg or not cfg['url']:
#        _logger.warning("Pretrained model URL is invalid, using random initialization.")
#        return

#    state_dict = model_zoo.load_url(cfg['url'], progress=False, map_location='cpu')

#    if filter_fn is not None:
#        state_dict = filter_fn(state_dict)

#    if in_chans == 1:
#        conv1_name = cfg['first_conv']
#        _logger.info('Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
#        conv1_weight = state_dict[conv1_name + '.weight']
#        # Some weights are in torch.half, ensure it's float for sum on CPU
#        conv1_type = conv1_weight.dtype
#        conv1_weight = conv1_weight.float()
#        O, I, J, K = conv1_weight.shape
#        if I > 3:
#            assert conv1_weight.shape[1] % 3 == 0
#            # For models with space2depth stems
#            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
#            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
#        else:
#            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
#        conv1_weight = conv1_weight.to(conv1_type)
#        state_dict[conv1_name + '.weight'] = conv1_weight
#    elif in_chans != 3:
#        conv1_name = cfg['first_conv']
#        conv1_weight = state_dict[conv1_name + '.weight']
#        conv1_type = conv1_weight.dtype
#        conv1_weight = conv1_weight.float()
#        O, I, J, K = conv1_weight.shape
#        if I != 3:
#            _logger.warning('Deleting first conv (%s) from pretrained weights.' % conv1_name)
#            del state_dict[conv1_name + '.weight']
#            strict = False
#        else:
#            # NOTE this strategy should be better than random init, but there could be other combinations of
#            # the original RGB input layer weights that'd work better for specific cases.
#            _logger.info('Repeating first conv (%s) weights in channel dim.' % conv1_name)
#            repeat = int(math.ceil(in_chans / 3))
#            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
#            conv1_weight *= (3 / float(in_chans))
#            conv1_weight = conv1_weight.to(conv1_type)
#            state_dict[conv1_name + '.weight'] = conv1_weight

#    classifier_name = cfg['classifier']
#    if num_classes == 1000 and cfg['num_classes'] == 1001:
#        # special case for imagenet trained models with extra background class in pretrained weights
#        classifier_weight = state_dict[classifier_name + '.weight']
#        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
#        classifier_bias = state_dict[classifier_name + '.bias']
#        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
#    elif num_classes != cfg['num_classes']:
#        # completely discard fully connected for all other differences between pretrained and created model
#        del state_dict[classifier_name + '.weight']
#        del state_dict[classifier_name + '.bias']
#        strict = False

#    model.load_state_dict(state_dict, strict=strict)


#def extract_layer(model, layer):
#    layer = layer.split('.')
#    module = model
#    if hasattr(model, 'module') and layer[0] != 'module':
#        module = model.module
#    if not hasattr(model, 'module') and layer[0] == 'module':
#        layer = layer[1:]
#    for l in layer:
#        if hasattr(module, l):
#            if not l.isdigit():
#                module = getattr(module, l)
#            else:
#                module = module[int(l)]
#        else:
#            return module
#    return module


#def set_layer(model, layer, val):
#    layer = layer.split('.')
#    module = model
#    if hasattr(model, 'module') and layer[0] != 'module':
#        module = model.module
#    lst_index = 0
#    module2 = module
#    for l in layer:
#        if hasattr(module2, l):
#            if not l.isdigit():
#                module2 = getattr(module2, l)
#            else:
#                module2 = module2[int(l)]
#            lst_index += 1
#    lst_index -= 1
#    for l in layer[:lst_index]:
#        if not l.isdigit():
#            module = getattr(module, l)
#        else:
#            module = module[int(l)]
#    l = layer[lst_index]
#    setattr(module, l, val)


#def adapt_model_from_string(parent_module, model_string):
#    separator = '***'
#    state_dict = {}
#    lst_shape = model_string.split(separator)
#    for k in lst_shape:
#        k = k.split(':')
#        key = k[0]
#        shape = k[1][1:-1].split(',')
#        if shape[0] != '':
#            state_dict[key] = [int(i) for i in shape]

#    new_module = deepcopy(parent_module)
#    for n, m in parent_module.named_modules():
#        old_module = extract_layer(parent_module, n)
#        if isinstance(old_module, nn.Conv2d) or isinstance(old_module, Conv2dSame):
#            if isinstance(old_module, Conv2dSame):
#                conv = Conv2dSame
#            else:
#                conv = nn.Conv2d
#            s = state_dict[n + '.weight']
#            in_channels = s[1]
#            out_channels = s[0]
#            g = 1
#            if old_module.groups > 1:
#                in_channels = out_channels
#                g = in_channels
#            new_conv = conv(
#                in_channels=in_channels, out_channels=out_channels, kernel_size=old_module.kernel_size,
#                bias=old_module.bias is not None, padding=old_module.padding, dilation=old_module.dilation,
#                groups=g, stride=old_module.stride)
#            set_layer(new_module, n, new_conv)
#        if isinstance(old_module, nn.BatchNorm2d):
#            new_bn = nn.BatchNorm2d(
#                num_features=state_dict[n + '.weight'][0], eps=old_module.eps, momentum=old_module.momentum,
#                affine=old_module.affine, track_running_stats=True)
#            set_layer(new_module, n, new_bn)
#        if isinstance(old_module, nn.Linear):
#            # FIXME extra checks to ensure this is actually the FC classifier layer and not a diff Linear layer?
#            num_features = state_dict[n + '.weight'][1]
#            new_fc = nn.Linear(
#                in_features=num_features, out_features=old_module.out_features, bias=old_module.bias is not None)
#            set_layer(new_module, n, new_fc)
#            if hasattr(new_module, 'num_features'):
#                new_module.num_features = num_features
#    new_module.eval()
#    parent_module.eval()

#    return new_module


#def adapt_model_from_file(parent_module, model_variant):
#    adapt_file = os.path.join(os.path.dirname(__file__), 'pruned', model_variant + '.txt')
#    with open(adapt_file, 'r') as f:
#        return adapt_model_from_string(parent_module, f.read().strip())


#class Callable:
#    def __init__(self, *args, **kwargs):
#        return super().__init__(*args, **kwargs)

#def build_model_with_cfg(
#        model_cls: Callable,
#        variant: str,
#        pretrained: bool,
#        default_cfg: dict,
#        model_cfg: dict = None,
#        feature_cfg: dict = None,
#        pretrained_strict: bool = True,
#        pretrained_filter_fn: Callable = None,
#        **kwargs):
#    pruned = kwargs.pop('pruned', False)
#    features = False
#    feature_cfg = feature_cfg or {}

#    if kwargs.pop('features_only', False):
#        features = True
#        feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
#        if 'out_indices' in kwargs:
#            feature_cfg['out_indices'] = kwargs.pop('out_indices')

#    model = model_cls(**kwargs) if model_cfg is None else model_cls(cfg=model_cfg, **kwargs)
#    model.default_cfg = deepcopy(default_cfg)
    
#    if pruned:
#        model = adapt_model_from_file(model, variant)

#    if pretrained:
#        load_pretrained(
#            model,
#            num_classes=kwargs.get('num_classes', 0),
#            in_chans=kwargs.get('in_chans', 3),
#            filter_fn=pretrained_filter_fn, strict=pretrained_strict)
    
#    if features:
#        feature_cls = FeatureListNet
#        if 'feature_cls' in feature_cfg:
#            feature_cls = feature_cfg.pop('feature_cls')
#            if isinstance(feature_cls, str):
#                feature_cls = feature_cls.lower()
#                if 'hook' in feature_cls:
#                    feature_cls = FeatureHookNet
#                else:
#                    assert False, f'Unknown feature class {feature_cls}'
#        model = feature_cls(model, **feature_cfg)
    
#    return model

####
#from timm.models.adaptive_avgmax_pool import SelectAdaptivePool2d
###

""" PyTorch selectable adaptive pooling
Adaptive pooling with the ability to select the type of pooling from:
    * 'avg' - Average pooling
    * 'max' - Max pooling
    * 'avgmax' - Sum of average and max pooling re-scaled by 0.5
    * 'avgmaxc' - Concatenation of average and max pooling along feature dim, doubles feature dim

Both a functional and a nn.Module version of the pooling is provided.

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type == 'catavgmax':
        return 2
    else:
        return 1


def adaptive_avgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)


def adaptive_catavgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return torch.cat((x_avg, x_max), 1)


def select_adaptive_pool2d(x, pool_type='avg', output_size=1):
    """Selectable global pooling function with dynamic input kernel size
    """
    if pool_type == 'avg':
        x = F.adaptive_avg_pool2d(x, output_size)
    elif pool_type == 'avgmax':
        x = adaptive_avgmax_pool2d(x, output_size)
    elif pool_type == 'catavgmax':
        x = adaptive_catavgmax_pool2d(x, output_size)
    elif pool_type == 'max':
        x = F.adaptive_max_pool2d(x, output_size)
    else:
        assert False, 'Invalid pool type: %s' % pool_type
    return x


class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(self, output_size=1):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_avgmax_pool2d(x, self.output_size)


class AdaptiveCatAvgMaxPool2d(nn.Module):
    def __init__(self, output_size=1):
        super(AdaptiveCatAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_catavgmax_pool2d(x, self.output_size)


class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(self, output_size=1, pool_type='avg', flatten=False):
        super(SelectAdaptivePool2d, self).__init__()
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        self.flatten = flatten
        if pool_type == '':
            self.pool = nn.Identity()  # pass through
        elif pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == 'avgmax':
            self.pool = AdaptiveAvgMaxPool2d(output_size)
        elif pool_type == 'catavgmax':
            self.pool = AdaptiveCatAvgMaxPool2d(output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            assert False, 'Invalid pool type: %s' % pool_type

    def is_identity(self):
        return self.pool_type == ''

    def forward(self, x):
        x = self.pool(x)
        if self.flatten:
            x = x.flatten(1)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'pool_type=' + self.pool_type \
               + ', flatten=' + str(self.flatten) + ')'


###



from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


__all__ = ['GenMUXNet']


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'muxnet_m': _cfg(
        url=''),
}


_DEBUG = False

# Default args for PyTorch BN impl
_BN_MOMENTUM_PT_DEFAULT = 0.1
_BN_EPS_PT_DEFAULT = 1e-5
_BN_ARGS_PT = dict(momentum=_BN_MOMENTUM_PT_DEFAULT, eps=_BN_EPS_PT_DEFAULT)

# Defaults used for Google/Tensorflow training of mobile networks /w RMSprop as per
# papers and TF reference implementations. PT momentum equiv for TF decay is (1 - TF decay)
# NOTE: momentum varies btw .99 and .9997 depending on source
# .99 in official TF TPU impl
# .9997 (/w .999 in search space) for paper
_BN_MOMENTUM_TF_DEFAULT = 1 - 0.99
_BN_EPS_TF_DEFAULT = 1e-3
_BN_ARGS_TF = dict(momentum=_BN_MOMENTUM_TF_DEFAULT, eps=_BN_EPS_TF_DEFAULT)


def _resolve_bn_args(kwargs):
    bn_args = _BN_ARGS_TF.copy() if kwargs.pop('bn_tf', False) else _BN_ARGS_PT.copy()
    bn_momentum = kwargs.pop('bn_momentum', None)
    if bn_momentum is not None:
        bn_args['momentum'] = bn_momentum
    bn_eps = kwargs.pop('bn_eps', None)
    if bn_eps is not None:
        bn_args['eps'] = bn_eps
    return bn_args


def _round_channels(channels, multiplier=1.0, divisor=8, channel_min=None):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return channels

    channels *= multiplier
    channel_min = channel_min or divisor
    new_channels = max(
        int(channels + divisor / 2) // divisor * divisor,
        channel_min)
    # Make sure that round down does not go down by more than 10%.
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return new_channels


def _parse_ksize(ss):
    if ss.isdigit():
        return int(ss)
    else:
        return [int(k) for k in ss.split('.')]


def _decode_block_str(block_str, depth_multiplier=1.0):
    """ Decode block definition string

    Gets a list of block arg (dicts) through a string notation of arguments.
    E.g. ir_r2_k3_s2_e1_i32_o16_se0.25_noskip

    All args can exist in any order with the exception of the leading string which
    is assumed to indicate the block type.

    leading string - block type (
      ir = InvertedResidual, ds = DepthwiseSep, dsa = DeptwhiseSep with pw act)
    r - number of repeat blocks,
    k - kernel size,
    s - strides (1-9),
    e - expansion ratio,
    c - output channels,
    se - squeeze/excitation ratio
    n - activation fn ('re', 'r6', 'hs', or 'sw')
    Args:
        block_str: a string representation of block arguments.
    Returns:
        A list of block args (dicts)
    Raises:
        ValueError: if the string def not properly specified (TODO)
    """
    assert isinstance(block_str, str)
    ops = block_str.split('_')
    block_type = ops[0]  # take the block type off the front
    ops = ops[1:]
    options = {}
    noskip = False
    for op in ops:
        # string options being checked on individual basis, combine if they grow
        if op == 'noskip':
            noskip = True
        elif op.startswith('n'):
            # activation fn
            key = op[0]
            v = op[1:]
            if v == 're':
                value = F.relu
            elif v == 'r6':
                value = F.relu6
            elif v == 'hs':
                value = hard_swish
            elif v == 'sw':
                value = swish
            else:
                continue
            options[key] = value
        else:
            # all numeric options
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

    # if act_fn is None, the model default (passed to model init) will be used
    act_fn = options['n'] if 'n' in options else None
    exp_kernel_size = _parse_ksize(options['a']) if 'a' in options else 1
    pw_kernel_size = _parse_ksize(options['p']) if 'p' in options else 1
    fake_in_chs = int(options['fc']) if 'fc' in options else 0  # FIXME hack to deal with in_chs issue in TPU def

    num_repeat = int(options['r'])
    # each type of block has different valid arguments, fill accordingly
    if block_type == 'ir':
        block_args = dict(
            block_type=block_type,
            dw_kernel_size=_parse_ksize(options['k']),
            exp_kernel_size=exp_kernel_size,
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options['c']),
            exp_ratio=float(options['e']),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            act_fn=act_fn,
            noskip=noskip,
        )
    elif block_type == 'sp':
        block_args = dict(
            block_type=block_type,
            dw_kernel_size=_parse_ksize(options['k']),
            exp_kernel_size=exp_kernel_size,
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options['c']),
            exp_ratio=float(options['e']),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            act_fn=act_fn,
            noskip=noskip,
        )
        if 'sr' in options: block_args['split_ratio'] = float(options['sr'])
        if 'sg' in options: block_args['shuffle_groups'] = int(options['sg'])
        if 'dwg' in options: block_args['dw_group_factor'] = int(options['dwg'])
        if 'sc' in options: block_args['scales'] = _parse_ksize(options['sc'])

    elif block_type == 'ds' or block_type == 'dsa':
        block_args = dict(
            block_type=block_type,
            dw_kernel_size=_parse_ksize(options['k']),
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options['c']),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            act_fn=act_fn,
            pw_act=block_type == 'dsa',
            noskip=block_type == 'dsa' or noskip,
        )
    else:
        assert False, 'Unknown block type (%s)' % block_type

    return block_args, num_repeat


def _scale_stage_depth(stack_args, repeats, depth_multiplier=1.0, depth_trunc='ceil'):
    """ Per-stage depth scaling
    Scales the block repeats in each stage. This depth scaling impl maintains
    compatibility with the EfficientNet scaling method, while allowing sensible
    scaling for other models that may have multiple block arg definitions in each stage.
    """

    # We scale the total repeat count for each stage, there may be multiple
    # block arg defs per stage so we need to sum.
    num_repeat = sum(repeats)
    if depth_trunc == 'round':
        # Truncating to int by rounding allows stages with few repeats to remain
        # proportionally smaller for longer. This is a good choice when stage definitions
        # include single repeat stages that we'd prefer to keep that way as long as possible
        num_repeat_scaled = max(1, round(num_repeat * depth_multiplier))
    else:
        # The default for EfficientNet truncates repeats to int via 'ceil'.
        # Any multiplier > 1.0 will result in an increased depth for every stage.
        num_repeat_scaled = int(math.ceil(num_repeat * depth_multiplier))

    # Proportionally distribute repeat count scaling to each block definition in the stage.
    # Allocation is done in reverse as it results in the first block being less likely to be scaled.
    # The first block makes less sense to repeat in most of the arch definitions.
    repeats_scaled = []
    for r in repeats[::-1]:
        rs = max(1, round((r / num_repeat * num_repeat_scaled)))
        repeats_scaled.append(rs)
        num_repeat -= r
        num_repeat_scaled -= rs
    repeats_scaled = repeats_scaled[::-1]

    # Apply the calculated scaling to each block arg in the stage
    sa_scaled = []
    for ba, rep in zip(stack_args, repeats_scaled):
        sa_scaled.extend([deepcopy(ba) for _ in range(rep)])
    return sa_scaled


def _decode_arch_def(arch_def, depth_multiplier=1.0, depth_trunc='ceil'):
    arch_args = []
    for stack_idx, block_strings in enumerate(arch_def):
        assert isinstance(block_strings, list)
        stack_args = []
        repeats = []
        for block_str in block_strings:
            assert isinstance(block_str, str)
            ba, rep = _decode_block_str(block_str)
            stack_args.append(ba)
            repeats.append(rep)
        arch_args.append(_scale_stage_depth(stack_args, repeats, depth_multiplier, depth_trunc))
    return arch_args


_USE_SWISH_OPT = True
if _USE_SWISH_OPT:
    class SwishAutoFn(torch.autograd.Function):
        """ Memory Efficient Swish
        From: https://blog.ceshine.net/post/pytorch-memory-swish/
        """
        @staticmethod
        def forward(ctx, x):
            result = x.mul(torch.sigmoid(x))
            ctx.save_for_backward(x)
            return result

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_variables[0]
            sigmoid_x = torch.sigmoid(x)
            return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


    def swish(x, inplace=False):
        return SwishAutoFn.apply(x)
else:
    def swish(x, inplace=False):
        return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


def sigmoid(x, inplace=False):
    return x.sigmoid_() if inplace else x.sigmoid()


def hard_swish(x, inplace=False):
    if inplace:
        return x.mul_(F.relu6(x + 3.) / 6.)
    else:
        return x * F.relu6(x + 3.) / 6.


def hard_sigmoid(x, inplace=False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class _BlockBuilder:
    """ Build Trunk Blocks

    This ended up being somewhat of a cross between
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py
    and
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_builder.py

    """
    def __init__(self, channel_multiplier=1.0, channel_divisor=8, channel_min=None,
                 pad_type='', act_fn=None, se_gate_fn=sigmoid, se_reduce_mid=False,
                 bn_args=_BN_ARGS_PT, drop_connect_rate=0., verbose=False):
        self.channel_multiplier = channel_multiplier
        self.channel_divisor = channel_divisor
        self.channel_min = channel_min
        self.pad_type = pad_type
        self.act_fn = act_fn
        self.se_gate_fn = se_gate_fn
        self.se_reduce_mid = se_reduce_mid
        self.bn_args = bn_args
        self.drop_connect_rate = drop_connect_rate
        self.verbose = verbose

        # updated during build
        self.in_chs = None
        self.block_idx = 0
        self.block_count = 0

    def _round_channels(self, chs):
        return _round_channels(chs, self.channel_multiplier, self.channel_divisor, self.channel_min)

    def _make_block(self, ba):
        bt = ba.pop('block_type')
        ba['in_chs'] = self.in_chs
        ba['out_chs'] = self._round_channels(ba['out_chs'])
        if 'fake_in_chs' in ba and ba['fake_in_chs']:
            # FIXME this is a hack to work around mismatch in origin impl input filters
            ba['fake_in_chs'] = self._round_channels(ba['fake_in_chs'])
        ba['bn_args'] = self.bn_args
        ba['pad_type'] = self.pad_type
        # block act fn overrides the model default
        ba['act_fn'] = ba['act_fn'] if ba['act_fn'] is not None else self.act_fn
        assert ba['act_fn'] is not None
        if bt == 'ir':
            ba['drop_connect_rate'] = self.drop_connect_rate * self.block_idx / self.block_count
            ba['se_gate_fn'] = self.se_gate_fn
            ba['se_reduce_mid'] = self.se_reduce_mid
            if self.verbose:
                logging.info('  InvertedResidual {}, Args: {}'.format(self.block_idx, str(ba)))
            block = InvertedResidual(**ba)
        elif bt == 'sp':
            ba['drop_connect_rate'] = self.drop_connect_rate * self.block_idx / self.block_count
            ba['se_gate_fn'] = self.se_gate_fn
            ba['se_reduce_mid'] = self.se_reduce_mid
            if self.verbose:
                logging.info('  MuxInvertedResidual {}, Args: {}'.format(self.block_idx, str(ba)))
            block = MuxInvertedResidual(**ba)
        elif bt == 'ds' or bt == 'dsa':
            ba['drop_connect_rate'] = self.drop_connect_rate * self.block_idx / self.block_count
            if self.verbose:
                logging.info('  DepthwiseSeparable {}, Args: {}'.format(self.block_idx, str(ba)))
            block = DepthwiseSeparableConv(**ba)
        else:
            assert False, 'Uknkown block type (%s) while building model.' % bt
        self.in_chs = ba['out_chs']  # update in_chs for arg of next block

        return block

    def _make_stack(self, stack_args):
        blocks = []
        # each stack (stage) contains a list of block arguments
        for i, ba in enumerate(stack_args):
            if self.verbose:
                logging.info(' Block: {}'.format(i))
            if i >= 1:
                # only the first block in any stack can have a stride > 1
                ba['stride'] = 1
            block = self._make_block(ba)
            blocks.append(block)
            self.block_idx += 1  # incr global idx (across all stacks)
        return nn.Sequential(*blocks)

    def __call__(self, in_chs, block_args):
        """ Build the blocks
        Args:
            in_chs: Number of input-channels passed to first block
            block_args: A list of lists, outer list defines stages, inner
                list contains strings defining block configuration(s)
        Return:
             List of block stacks (each stack wrapped in nn.Sequential)
        """
        if self.verbose:
            logging.info('Building model trunk with %d stages...' % len(block_args))
        self.in_chs = in_chs
        self.block_count = sum([len(x) for x in block_args])
        self.block_idx = 0
        blocks = []
        # outer list of block_args defines the stacks ('stages' by some conventions)
        for stack_idx, stack in enumerate(block_args):
            if self.verbose:
                logging.info('Stack: {}'.format(stack_idx))
            assert isinstance(stack, list)
            stack = self._make_stack(stack)
            blocks.append(stack)
        return blocks


def _initialize_weight_goog(m):
    # weight init as per Tensorflow Official impl
    # https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # fan-out
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(0)  # fan-out
        init_range = 1.0 / math.sqrt(n)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


def _initialize_weight_default(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='linear')


def drop_connect(inputs, training=False, drop_connect_rate=0.):
    """Apply drop connect."""
    if not training:
        return inputs

    keep_prob = 1 - drop_connect_rate
    random_tensor = keep_prob + torch.rand(
        (inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
    random_tensor.floor_()  # binarize
    output = inputs.div(keep_prob) * random_tensor
    return output


class ChannelShuffle(nn.Module):
    # FIXME haven't used yet
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def __repr__(self):
        return '%s(groups=%d)' % (self.__class__.__name__, self.groups)

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, "Incompatible group size {} for input channel {}".format(
            g, C
        )
        return (
            x.view(N, g, int(C / g), H, W)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(N, C, H, W)
        )


class SplitBlock(nn.Module):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def __repr__(self):
        return '%s(ratio=%.2f)' % (self.__class__.__name__, self.ratio)

    def forward(self, x):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, reduce_chs=None, act_fn=F.relu, gate_fn=sigmoid):
        super(SqueezeExcite, self).__init__()
        self.act_fn = act_fn
        self.gate_fn = gate_fn
        reduced_chs = reduce_chs or in_chs
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        # NOTE adaptiveavgpool can be used here, but seems to cause issues with NVIDIA AMP performance
        x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.conv_reduce(x_se)
        x_se = self.act_fn(x_se, inplace=True)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks with an expansion
    factor of 1.0. This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, pad_type='', act_fn=F.relu, noskip=False,
                 pw_kernel_size=1, pw_act=False,
                 se_ratio=0., se_gate_fn=sigmoid,
                 bn_args=_BN_ARGS_PT, drop_connect_rate=0.):
        super(DepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv
        self.act_fn = act_fn
        self.drop_connect_rate = drop_connect_rate

        self.conv_dw = select_conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, padding=pad_type, depthwise=True)
        self.bn1 = nn.BatchNorm2d(in_chs, **bn_args)

        # Squeeze-and-excitation
        if self.has_se:
            self.se = SqueezeExcite(
                in_chs, reduce_chs=max(1, int(in_chs * se_ratio)), act_fn=act_fn, gate_fn=se_gate_fn)

        self.conv_pw = select_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = nn.BatchNorm2d(out_chs, **bn_args)

    def forward(self, x):
        residual = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)

        if self.has_se:
            x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_pw_act:
            x = self.act_fn(x, inplace=True)

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual
        return x


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, pad_type='', act_fn=F.relu, noskip=False,
                 exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1,
                 se_ratio=0., se_reduce_mid=False, se_gate_fn=sigmoid,
                 shuffle_type=None, bn_args=_BN_ARGS_PT, drop_connect_rate=0.):
        super(InvertedResidual, self).__init__()
        mid_chs = int(in_chs * exp_ratio)
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.act_fn = act_fn
        self.drop_connect_rate = drop_connect_rate

        # Point-wise expansion
        self.conv_pw = select_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type)
        self.bn1 = nn.BatchNorm2d(mid_chs, **bn_args)

        self.shuffle_type = shuffle_type
        if shuffle_type is not None and isinstance(exp_kernel_size, list):
            self.shuffle = ChannelShuffle(len(exp_kernel_size))

        # Depth-wise convolution
        self.conv_dw = select_conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, padding=pad_type, depthwise=True)
        self.bn2 = nn.BatchNorm2d(mid_chs, **bn_args)

        # Squeeze-and-excitation
        if self.has_se:
            se_base_chs = mid_chs if se_reduce_mid else in_chs
            self.se = SqueezeExcite(
                mid_chs, reduce_chs=max(1, int(se_base_chs * se_ratio)), act_fn=act_fn, gate_fn=se_gate_fn)

        # Point-wise linear projection
        self.conv_pwl = select_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn3 = nn.BatchNorm2d(out_chs, **bn_args)

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)

        # FIXME haven't tried this yet
        # for channel shuffle when using groups with pointwise convs as per FBNet variants
        if self.shuffle_type == "mid":
            x = self.shuffle(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act_fn(x, inplace=True)

        # Squeeze-and-excitation
        if self.has_se:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual

        # NOTE maskrcnn_benchmark building blocks have an SE module defined here for some variants

        return x


class MuxInvertedResidual(nn.Module):
    """ Inverted residual block w/ Channel Shuffling w/ optional SE"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, pad_type='', act_fn=F.relu, noskip=False,
                 exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1,
                 se_ratio=0., se_reduce_mid=False, se_gate_fn=sigmoid,
                 shuffle_type=None, bn_args=_BN_ARGS_PT, drop_connect_rate=0.,
                 split_ratio=0.75, shuffle_groups=2, dw_group_factor=1,
                 scales=0):
        super(MuxInvertedResidual, self).__init__()

        assert in_chs == out_chs, "should only be used when input channels == output channels"
        assert stride < 2, "should NOT be used to down-sample"

        self.split = SplitBlock(split_ratio)
        in_chs = int(in_chs * split_ratio)
        out_chs = int(out_chs * split_ratio)
        mid_chs = int(in_chs * exp_ratio)

        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.act_fn = act_fn
        self.drop_connect_rate = drop_connect_rate

        # Point-wise expansion
        self.conv_pw = select_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type)
        self.bn1 = nn.BatchNorm2d(mid_chs, **bn_args)

        # Depth-wise/group-wise convolution
        self.conv_dw = select_conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, padding=pad_type,
            groups=mid_chs // dw_group_factor,
            scales=scales
        )
        self.bn2 = nn.BatchNorm2d(mid_chs, **bn_args)

        # Squeeze-and-excitation
        if self.has_se:
            se_base_chs = mid_chs if se_reduce_mid else in_chs
            self.se = SqueezeExcite(
                mid_chs, reduce_chs=max(1, int(se_base_chs * se_ratio)), act_fn=act_fn, gate_fn=se_gate_fn)

        # Point-wise linear projection
        self.conv_pwl = select_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn3 = nn.BatchNorm2d(out_chs, **bn_args)

        self.shuffle = ChannelShuffle(groups=shuffle_groups)

    def forward(self, x):

        x, x1 = self.split(x)

        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act_fn(x, inplace=True)

        # Squeeze-and-excitation
        if self.has_se:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            # if self.drop_connect_rate > 0.:
            #     x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual

        x = torch.cat([x, x1], dim=1)
        x = self.shuffle(x)

        return x


class GenMUXNet(nn.Module):
    """ Generic MUXNet"""

    def __init__(self, block_args, num_classes=1000, in_chans=3, stem_size=32, num_features=1280,
                 channel_multiplier=1.0, channel_divisor=8, channel_min=None,
                 pad_type='', act_fn=F.relu, drop_rate=0., drop_connect_rate=0.,
                 se_gate_fn=sigmoid, se_reduce_mid=False, bn_args=_BN_ARGS_PT,
                 global_pool='avg', head_conv='default', weight_init='goog'):
        super(GenMUXNet, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.act_fn = act_fn
        self.num_features = num_features

        stem_size = _round_channels(stem_size, channel_multiplier, channel_divisor, channel_min)
        self.conv_stem = select_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = nn.BatchNorm2d(stem_size, **bn_args)
        in_chs = stem_size

        builder = _BlockBuilder(
            channel_multiplier, channel_divisor, channel_min,
            pad_type, act_fn, se_gate_fn, se_reduce_mid,
            bn_args, drop_connect_rate, verbose=_DEBUG)
        self.blocks = nn.Sequential(*builder(in_chs, block_args))
        in_chs = builder.in_chs

        if not head_conv or head_conv == 'none':
            self.efficient_head = False
            self.conv_head = None
            assert in_chs == self.num_features
        else:
            self.efficient_head = head_conv == 'efficient'
            self.conv_head = select_conv2d(in_chs, self.num_features, 1, padding=pad_type)
            self.bn2 = None if self.efficient_head else nn.BatchNorm2d(self.num_features, **bn_args)

        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.classifier = nn.Linear(self.num_features * self.global_pool.feat_mult(), self.num_classes)

        for m in self.modules():
            if weight_init == 'goog':
                _initialize_weight_goog(m)
            else:
                _initialize_weight_default(m)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        del self.classifier
        if num_classes:
            self.classifier = nn.Linear(
                self.num_features * self.global_pool.feat_mult(), num_classes)
        else:
            self.classifier = None

    def forward_features(self, x, pool=True):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)
        x = self.blocks(x)
        if self.efficient_head:
            # efficient head, currently only mobilenet-v3 performs pool before last 1x1 conv
            x = self.global_pool(x)  # always need to pool here regardless of flag
            x = self.conv_head(x)
            # no BN
            x = self.act_fn(x, inplace=True)
            if pool:
                # expect flattened output if pool is true, otherwise keep dim
                x = x.view(x.size(0), -1)
        else:
            if self.conv_head is not None:
                x = self.conv_head(x)
                x = self.bn2(x)
            x = self.act_fn(x, inplace=True)
            if pool:
                x = self.global_pool(x)
                x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)


def _gen_muxnet_m(channel_multiplier=1.0, depth_multiplier=1.0, num_classes=1000, **kwargs):
    """Creates an MUXNet-m model.

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage

    """
    num_features = _round_channels(1280, channel_multiplier, 8, None)
    arch_def = [
        ['ds_r1_k3_s1_e1_c24_se0.25_nre'],

        ['ir_r1_k3_s2_e4_c24_se0.75_nre'],
        ['sp_r1_k3_s1_e4_c24_se0.75_sr0.5_sg2_dwg2_nre'],
        ['sp_r1_k3_s1_e4_c24_se0.75_sr0.5_sg2_dwg2_nre'],

        ['ir_r1_k3.5.7_s2_e4_c40_se0.75'],
        ['sp_r1_k3_s1_e6_c40_se0.75_sr0.5_sg2_dwg2'],
        ['sp_r1_k3_s1_e6_c40_se0.75_sr0.5_sg2_dwg2'],

        ['ir_r1_k3.5.7.9_s2_e4_c80_se0.75'],
        ['sp_r1_k5_s1_e6_c80_se0.75_sr0.5_sg2_dwg2'],
        ['sp_r1_k5_s1_e6_c80_se0.75_sr0.5_sg2_dwg2'],

        ['ir_r1_k5_s1_e6_c112_se0.75'],
        ['sp_r1_k5_s1_e6_c112_se0.75_sr0.5_sg2_dwg2'],
        ['sp_r1_k5_s1_e6_c112_se0.75_sr0.5_sg2_dwg2'],

        ['ir_r1_k3.5.7.9.11_s2_e4_c160_se0.75'],
        ['sp_r1_k5_s1_e6_c160_se0.75_sr0.5_sg2_dwg2'],
        ['sp_r1_k5_s1_e6_c160_se0.75_sr0.5_sg2_dwg2'],
        ['sp_r1_k5_s1_e6_c160_se0.75_sr0.5_sg2_dwg2'],

        ['ir_r1_k3_s1_e6_c200_se0.75'],
    ]
    model = GenMUXNet(
        _decode_arch_def(arch_def, depth_multiplier),
        num_classes=num_classes,
        stem_size=24,
        channel_multiplier=channel_multiplier,
        num_features=num_features,
        bn_args=_resolve_bn_args(kwargs),
        act_fn=swish,
        **kwargs
    )
    return model


def _gen_muxnet_l(channel_multiplier=1.0, depth_multiplier=1.0, num_classes=1000, **kwargs):
    """Creates an MUXNet-l model.

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage

    """
    num_features = _round_channels(1536, channel_multiplier, 8, None)
    arch_def = [
        ['ds_r1_k3_s1_e1_c24_se0.25_nre'],

        ['ir_r1_k3_s2_e6_c24_se0.75_nre'],
        ['sp_r1_k5_s1_e6_c24_se0.75_sr0.5_sg2_dwg2_nre'],
        ['sp_r1_k5_s1_e6_c24_se0.75_sr0.5_sg2_dwg2_nre'],

        ['ir_r1_k3.5.7_s2_e6_c40_se0.75'],
        ['sp_r1_k5_s1_e6_c40_se0.75_sr0.5_sg2_dwg2'],
        ['sp_r1_k5_s1_e6_c40_se0.75_sr0.5_sg2_dwg2'],
        ['sp_r1_k5_s1_e6_c40_se0.75_sr0.5_sg2_dwg2'],

        ['ir_r1_k3.5.7.9_s2_e6_c80_se0.75'],
        ['sp_r1_k5_s1_e6_c80_se0.75_sr0.5_sg2_dwg2'],
        ['sp_r1_k5_s1_e6_c80_se0.75_sr0.5_sg2_dwg2'],
        ['sp_r1_k5_s1_e6_c80_se0.75_sr0.5_sg2_dwg2'],

        ['ir_r1_k5_s1_e6_c120_se0.75'],
        ['sp_r1_k5_s1_e6_c120_se0.75_sr0.5_sg2_dwg2'],
        ['sp_r1_k5_s1_e6_c120_se0.75_sr0.5_sg2_dwg2'],
        ['sp_r1_k5_s1_e6_c120_se0.75_sr0.5_sg2_dwg2'],

        ['ir_r1_k3.5.7.9.11_s2_e6_c160_se0.75'],
        ['sp_r1_k5_s1_e6_c160_se0.75_sr0.5_sg2_dwg2'],
        ['sp_r1_k5_s1_e6_c160_se0.75_sr0.5_sg2_dwg2'],
        ['sp_r1_k5_s1_e6_c160_se0.75_sr0.5_sg2_dwg2'],

        ['ir_r1_k3_s1_e6_c200_se0.75'],
    ]
    model = GenMUXNet(
        _decode_arch_def(arch_def, depth_multiplier),
        num_classes=num_classes,
        stem_size=24,
        channel_multiplier=channel_multiplier,
        num_features=num_features,
        bn_args=_resolve_bn_args(kwargs),
        act_fn=swish,
        **kwargs
    )
    return model


@register_model
def muxnet_m(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ MUXNet-m """
    default_cfg = default_cfgs['muxnet_m']
    # NOTE for train, drop_rate should be 0.2
    kwargs['drop_connect_rate'] = 0.1  # set when training, TODO add as cmd arg
    model = _gen_muxnet_m(
        channel_multiplier=1.0, depth_multiplier=1.0,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        checkpoint_path = 'muxnet_m.init'
        load_checkpoint(model, checkpoint_path, use_ema=True)
    return model


@register_model
def muxnet_l(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ MUXNet-l """
    default_cfg = default_cfgs['msunet_s']
    # NOTE for train, drop_rate should be 0.2
    kwargs['drop_connect_rate'] = 0.15  # set when training, TODO add as cmd arg
    model = _gen_muxnet_l(
        channel_multiplier=1.0, depth_multiplier=1.0,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def factory(name, **kwargs):
    if name == 'muxnet_m':
        return muxnet_m(**kwargs)
    else:
        raise NotImplementedError("Unknown model requested")


if __name__ == '__main__':
    import warnings
    from torchprofile import profile_macs
    warnings.filterwarnings("ignore")

    model = muxnet_m(pretrained=False)
    inputs = torch.randn(1, 3, 224, 224)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops = profile_macs(model, inputs)

    print('Total params: %.2fM' % (params / 1000000.0))
    print('Total flops: %.2fM' % (flops / 1000000.0))