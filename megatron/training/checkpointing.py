# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Input/output checkpointing."""

from logging import getLogger
import os
import random
import sys
import numpy as np
from time import time

import torch

from megatron.core import mpu, tensor_parallel, dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
from megatron.core.dist_checkpointing.strategies.fully_parallel import \
    FullyParallelSaveStrategyWrapper, FullyParallelLoadStrategyWrapper
from megatron.core.num_microbatches_calculator import update_num_microbatches
from .async_utils import schedule_async_save
from .global_vars import get_args, get_one_logger
from .utils import unwrap_model, print_rank_0, append_to_progress_log, is_last_rank
from ..core.dist_checkpointing.serialization import \
    get_default_save_sharded_strategy
from .one_logger_utils import on_save_checkpoint_start, on_save_checkpoint_success

# [ModelOpt]: Import
try:
    from modelopt.torch.opt.plugins import (
        save_modelopt_state,
        save_sharded_modelopt_state,
        restore_modelopt_state,
        restore_sharded_modelopt_state,
    )
    has_nvidia_modelopt = True
except Exception:
    has_nvidia_modelopt = False

_CHECKPOINT_VERSION = None

logger = getLogger(__name__)

def set_checkpoint_version(value):
    global _CHECKPOINT_VERSION
    if _CHECKPOINT_VERSION is not None:
        assert _CHECKPOINT_VERSION == value, \
            "checkpoint versions do not match"
    _CHECKPOINT_VERSION = value


def get_checkpoint_version():
    global _CHECKPOINT_VERSION
    return _CHECKPOINT_VERSION


def check_checkpoint_args(checkpoint_args):
    """Ensure fixed arguments for a model are the same for the input
    arguments and the one retrieved from checkpoint."""
    args = get_args()

    def _compare(arg_name, old_arg_name=None, default=None):
        if old_arg_name is not None:
            ckpt_arg_name = old_arg_name
        else:
            ckpt_arg_name = arg_name
        if default is not None:
            checkpoint_value = getattr(checkpoint_args, ckpt_arg_name, default)
        else:
            checkpoint_value = getattr(checkpoint_args, ckpt_arg_name)
        args_value = getattr(args, arg_name)
        error_message = '{} value from checkpoint ({}) is not equal to the ' \
                        'input argument value ({}).'.format(
                            arg_name, checkpoint_value, args_value)
        assert checkpoint_value == args_value, error_message

    _compare('num_layers')
    _compare('hidden_size')
    _compare('num_attention_heads')
    _compare('add_position_embedding', default=True)
    if args.vocab_file:
        _compare('max_position_embeddings')
        _compare('make_vocab_size_divisible_by')
        if not args.use_dist_ckpt:
            _compare('padded_vocab_size')
        _compare('tokenizer_type')
    if args.data_parallel_random_init:
        _compare('data_parallel_random_init')
    if get_checkpoint_version() < 3.0:
        _compare('tensor_model_parallel_size',
                 old_arg_name='model_parallel_size')
    if get_checkpoint_version() >= 3.0 and not args.use_dist_ckpt:
        _compare('tensor_model_parallel_size')
        _compare('pipeline_model_parallel_size')

def ensure_directory_exists(filename, check_parent=True):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename) if check_parent else filename
    os.makedirs(dirname, exist_ok=True)


def get_checkpoint_name(checkpoints_path, iteration, release=False,
                        pipeline_parallel=None,
                        tensor_rank=None, pipeline_rank=None,
                        expert_parallel=None, expert_rank=None,
                        return_base_dir=False):
    """Determine the directory name for this rank's checkpoint."""
    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)
    if return_base_dir:
        common_path = os.path.join(checkpoints_path, directory)
        return common_path

    # Use both the tensor and pipeline MP rank.
    if pipeline_parallel is None:
        pipeline_parallel = (mpu.get_pipeline_model_parallel_world_size() > 1)
    if tensor_rank is None:
        tensor_rank = mpu.get_tensor_model_parallel_rank()
    if pipeline_rank is None:
        pipeline_rank = mpu.get_pipeline_model_parallel_rank()
    if expert_parallel is None:
        expert_parallel = (mpu.get_expert_model_parallel_world_size() > 1)
    if expert_rank is None:
        expert_rank = mpu.get_expert_model_parallel_rank()

    # Use both the tensor and pipeline MP rank. If using the distributed
    # optimizer, then the optimizer's path must additionally include the
    # data parallel rank.
    if not pipeline_parallel:
        common_path = os.path.join(checkpoints_path, directory,
                            f'mp_rank_{tensor_rank:02d}')
    else:
        common_path = os.path.join(checkpoints_path, directory,
                f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}')

    if expert_parallel:
        common_path = common_path + f'_{expert_rank:03d}'

    return os.path.join(common_path, "model_optim_rng.pt")


def get_distributed_optimizer_checkpoint_name(model_checkpoint_name):
    return os.path.join(os.path.dirname(model_checkpoint_name),
                        "distrib_optim.pt")


def find_checkpoint_rank_0(checkpoints_path, iteration, release=False):
    """Finds the checkpoint for rank 0 without knowing if we are using
    pipeline parallelism/expert parallelism or not.

    Since the checkpoint naming scheme changes if pipeline or expert
    parallelism is present, we need to look for both naming schemes if
    we don't know if the checkpoint has pipeline or expert parallelism.
    """

    # Look for checkpoint with no pipelining and no expert parallelism
    filename = get_checkpoint_name(checkpoints_path, iteration, release,
                                   pipeline_parallel=False,
                                   tensor_rank=0, pipeline_rank=0,
                                   expert_parallel=False, expert_rank=0)
    if os.path.isfile(filename):
        return filename

    # Look for checkpoint with no pipelining and expert parallelism
    filename = get_checkpoint_name(checkpoints_path, iteration, release,
                                   pipeline_parallel=False,
                                   tensor_rank=0, pipeline_rank=0,
                                   expert_parallel=True, expert_rank=0)
    if os.path.isfile(filename):
        return filename

    # Look for checkpoint with pipelining and no expert parallelism
    filename = get_checkpoint_name(checkpoints_path, iteration, release,
                                   pipeline_parallel=True,
                                   tensor_rank=0, pipeline_rank=0,
                                   expert_parallel=False, expert_rank=0)
    if os.path.isfile(filename):
        return filename

    # Look for checkpoint with pipelining and expert parallelism
    filename = get_checkpoint_name(checkpoints_path, iteration, release,
                                   pipeline_parallel=True,
                                   tensor_rank=0, pipeline_rank=0,
                                   expert_parallel=True, expert_rank=0)
    if os.path.isfile(filename):
        return filename

    # Look for a distributed checkpoint
    filename = get_checkpoint_name(checkpoints_path, iteration, release,
                                   pipeline_parallel=True,
                                   return_base_dir=True)
    if dist_checkpointing.check_is_distributed_checkpoint(filename):
        return filename

    return None


def get_checkpoint_tracker_filename(checkpoints_path):

    """Tracker file rescords the latest chckpoint during
    training to restart from."""
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')


def checkpoint_exists(checkpoints_path):
    if checkpoints_path is None:
        return False
    load_step = 'latest_checkpointed_iteration.txt'
    return os.path.exists(os.path.join(checkpoints_path, load_step))


def read_metadata(tracker_filename):
    # Read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration = 0
    release = False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                print_rank_0('ERROR: Invalid metadata file {}. Exiting'.format(
                    tracker_filename))
                sys.exit()
    assert iteration > 0 or release, 'error parsing metadata file {}'.format(
        tracker_filename)

    # Get the max iteration retrieved across the ranks.
    if torch.distributed.is_initialized():
        iters_cuda = torch.tensor([iteration], dtype=torch.long, device='cuda')
        torch.distributed.all_reduce(iters_cuda, op=torch.distributed.ReduceOp.MAX)
        max_iter = iters_cuda[0].item()

        # We should now have all the same iteration.
        # If not, print a warning and chose the maximum
        # iteration across all ranks.
        if iteration != max_iter:
            rank = torch.distributed.get_rank()
            print('WARNING: on rank {} found iteration {} in the '
                  'metadata while max iteration across the ranks '
                  'is {}, replacing it with max iteration.'.format(
                      rank, iteration, max_iter), flush=True)
    else:
        # When loading a checkpoint outside of training (for example,
        # when editing it), we might not have torch distributed
        # initialized, in this case, just assume we have the latest
        max_iter = iteration
    return max_iter, release


def get_rng_state(use_dist_ckpt: bool = False):
    """ collect rng state across data parallel ranks """
    args = get_args()
    rng_state = {
        'random_rng_state': random.getstate(),
        'np_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state(),
        'rng_tracker_states': tensor_parallel.get_cuda_rng_tracker().get_states()}

    rng_state_list = None
    if torch.distributed.is_initialized() and \
            mpu.get_data_parallel_world_size() > 1 and \
            args.data_parallel_random_init:
        rng_state_list = \
            [None for i in range(mpu.get_data_parallel_world_size())]
        torch.distributed.all_gather_object(
            rng_state_list,
            rng_state,
            group=mpu.get_data_parallel_group())
    else:
        rng_state_list = [rng_state]

    if use_dist_ckpt:
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        rng_state_list = ShardedObject('rng_state', rng_state_list, (pp_size, tp_size), (pp_rank, tp_rank),
                                       replica_id=mpu.get_data_parallel_rank(with_context_parallel=True))

    return rng_state_list


def save_checkpoint(iteration, model, optimizer, opt_param_scheduler, num_floating_point_operations_so_far, checkpointing_context=None,
                    pipeline_rank=None,expert_rank=None, tensor_rank=None, pipeline_parallel=None, expert_parallel=None):
    """Save a model checkpoint.

    Checkpointing context is used to persist some checkpointing state
    throughout a single job. Must be initialized externally (not used if None).
    """
    start_ckpt = time()
    args = get_args()

    # Prepare E2E metrics at start of save checkpoint
    productive_metrics = on_save_checkpoint_start(args.async_save)

    # Only rank zero of the data parallel writes to the disk.
    model = unwrap_model(model)

    ckpt_format = args.dist_ckpt_format if args.use_dist_ckpt else 'torch'
    print_rank_0('saving checkpoint at iteration {:7d} to {} in {} format'.format(
        iteration, args.save, ckpt_format))

    # Collect rng state across data parallel ranks.
    rng_state = get_rng_state(args.use_dist_ckpt)

    # Checkpoint name.
    checkpoint_name = get_checkpoint_name(args.save, iteration, release=False, pipeline_parallel=pipeline_parallel,
        tensor_rank=tensor_rank, pipeline_rank=pipeline_rank, expert_parallel=expert_parallel, expert_rank=expert_rank, return_base_dir=args.use_dist_ckpt)

    # Save distributed optimizer's custom parameter state.
    if args.use_distributed_optimizer and not args.no_save_optim and optimizer is not None and not args.use_dist_ckpt:
        optim_checkpoint_name = \
            get_distributed_optimizer_checkpoint_name(checkpoint_name)
        ensure_directory_exists(optim_checkpoint_name)
        optimizer.save_parameter_state(optim_checkpoint_name)

    async_save_request = None
    if args.async_save:
        if not args.use_dist_ckpt:
            raise NotImplementedError('Async checkpoint save not implemented for legacy checkpoints')
        elif args.dist_ckpt_format != 'torch_dist':
            raise NotImplementedError(f'Async checkpoint save not implemented for {args.dist_ckpt_format} distributed checkpoint format')

    # Collect args, model, RNG.
    if not torch.distributed.is_initialized() \
            or mpu.get_data_modulo_expert_parallel_rank(with_context_parallel=True) == 0 \
            or args.use_dist_ckpt:

        optim_sd_kwargs = {}
        if args.use_dist_ckpt and args.use_distributed_optimizer:
            optim_sd_kwargs['sharding_type'] = ('fully_sharded_model_space'
                                                if args.ckpt_fully_parallel_save
                                                else 'dp_zero_gather_scatter')
            print_rank_0(f'Storing distributed optimizer sharded state of type {optim_sd_kwargs["sharding_type"]}')
        state_dict = generate_state_dict(args, model, optimizer, opt_param_scheduler, rng_state,
                                         args.use_dist_ckpt, iteration, optim_sd_kwargs=optim_sd_kwargs)

        state_dict['num_floating_point_operations_so_far'] = num_floating_point_operations_so_far
        if args.use_dist_ckpt:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                ensure_directory_exists(checkpoint_name, check_parent=False)
            validate_sharding_integrity = True
            save_strategy = (checkpointing_context or {}).get('save_strategy',
                                                              get_default_save_sharded_strategy(args.dist_ckpt_format))
            if args.ckpt_assume_constant_structure and args.dist_ckpt_format == 'torch_dist':
                save_strategy.use_cached_ckpt_structure = args.ckpt_assume_constant_structure
            if args.ckpt_fully_parallel_save:
                if checkpointing_context is not None and 'save_strategy' in checkpointing_context:
                    # Already saved once before - don't need to rerun sharding validation
                    validate_sharding_integrity = not args.ckpt_assume_constant_structure
                else:
                    save_strategy = FullyParallelSaveStrategyWrapper(save_strategy, mpu.get_data_parallel_group(with_context_parallel=True),
                                                                     args.ckpt_assume_constant_structure)
            # Store save strategy for future checkpoint saves
            if checkpointing_context is not None:
                checkpointing_context['save_strategy'] = save_strategy
            end_ckpt = time()
            logger.debug(f"rank: {torch.distributed.get_rank()}, takes {end_ckpt - start_ckpt} to prepare state dict for ckpt ")
            async_save_request = dist_checkpointing.save(state_dict, checkpoint_name, save_strategy,
                                                         async_sharded_save=args.async_save)

            # [ModelOpt]: save sharded modelopt_state
            if has_nvidia_modelopt:
                save_sharded_modelopt_state(model, checkpoint_name, (args.dist_ckpt_format, 1))
        else:
            # [ModelOpt]: Inject modelopt_state into state_dict
            if has_nvidia_modelopt:
                save_modelopt_state(model, state_dict)

            # Save.
            ensure_directory_exists(checkpoint_name)
            torch.save(state_dict, checkpoint_name)
    start_misc = time()
    if not args.async_save:
        assert async_save_request is None
        # Wait so everyone is done (necessary)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    # And update the latest iteration
    if not torch.distributed.is_initialized() \
       or torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)

        def iter_finalize_fn():
            with open(tracker_filename, 'w') as f:
                f.write(str(iteration))
            print_rank_0('  successfully saved checkpoint from iteration {:7d} to {}'
                         .format(iteration, args.save))
            if args.log_progress and args.async_save:
                append_to_progress_log(f'Saved async checkpoint\tIteration: {iteration}',
                                       barrier=False)

        if args.async_save:
            assert async_save_request is not None
            async_save_request.add_finalize_fn(iter_finalize_fn)
        else:
            iter_finalize_fn()

    # Additional callback for one_logger (last rank)
    if not torch.distributed.is_initialized() \
       or is_last_rank():
        def onelogger_finalize_fn():
            on_save_checkpoint_success(productive_metrics, args.async_save)
        if args.async_save:
            assert async_save_request is not None
            async_save_request.add_finalize_fn(onelogger_finalize_fn)
        else:
            onelogger_finalize_fn()

    if args.async_save:
        schedule_async_save(async_save_request)
        print_rank_0('  scheduled an async checkpoint save at iteration {:7d} to {}' \
                     .format(iteration, args.save))

    # Wait so everyone is done (not necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    end_misc = time()
    logger.debug(f"rank: {torch.distributed.get_rank()}, takes {end_misc - start_misc} to finalize ckpt save ")

def generate_state_dict(args, model, optimizer, opt_param_scheduler,
                        rng_state, use_dist_ckpt=False, iteration=None,
                        optim_sd_kwargs=None):
    # Arguments, iteration, and model.
    state_dict = {}
    state_dict['args'] = args
    state_dict['checkpoint_version'] = 3.0
    if iteration is not None:
        state_dict['iteration'] = iteration

    if len(model) == 1:
        state_dict['model'] = (model[0].sharded_state_dict()
                               if use_dist_ckpt else
                               model[0].state_dict_for_save_checkpoint())
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            state_dict['model%d' % i] = (
                model[i].sharded_state_dict()
                if use_dist_ckpt else
                model[i].state_dict_for_save_checkpoint())
    # Optimizer stuff.
    if not args.no_save_optim:
        if optimizer is not None:
            state_dict['optimizer'] = (optimizer.sharded_state_dict(state_dict, **(optim_sd_kwargs or {}))
                                       if use_dist_ckpt else
                                       optimizer.state_dict())
        if opt_param_scheduler is not None:
            state_dict['opt_param_scheduler'] = \
                opt_param_scheduler.state_dict()
    # RNG states.
    if not args.no_save_rng:
        state_dict["rng_state"] = rng_state
    return state_dict


def _transpose_first_dim(t, num_splits, num_splits_first, model):
    input_shape = t.size()
    # We use a self_attention module but the values extracted aren't
    # specific to self attention so should work for cross attention as well
    while hasattr(model, 'module'):
        model = model.module
    attention_module = model.language_model.encoder.layers[0].self_attention
    hidden_size_per_attention_head = attention_module.hidden_size_per_attention_head
    num_attention_heads_per_partition = attention_module.num_attention_heads_per_partition
    if num_splits_first:
        """[num_splits * np * hn, h]
        -->(view) [num_splits, np, hn, h]
        -->(tranpose) [np, num_splits, hn, h]
        -->(view) [np * num_splits * hn, h] """

        intermediate_shape = \
            (num_splits, num_attention_heads_per_partition,
             hidden_size_per_attention_head) + input_shape[1:]

        t = t.view(*intermediate_shape)
        t = t.transpose(0, 1).contiguous()
    else:
        """[np * hn * num_splits, h]
        -->(view) [np, hn, num_splits, h]
        -->(tranpose) [np, num_splits, hn, h]
        -->(view) [np * num_splits * hn, h] """

        intermediate_shape = \
            (num_attention_heads_per_partition,
             hidden_size_per_attention_head, num_splits) +\
             input_shape[1:]

        t = t.view(*intermediate_shape)
        t = t.transpose(1, 2).contiguous()
    t = t.view(*input_shape)

    return t


def fix_query_key_value_ordering(model, checkpoint_version):
    """Fix up query/key/value matrix ordering if checkpoint
    version is smaller than 2.0
    """
    if checkpoint_version < 2.0:
        if isinstance(model, list):
            assert len(model)==1
            model = model[0]
        for name, param in model.named_parameters():
            if name.endswith(('.query_key_value.weight', '.query_key_value.bias')):
                if checkpoint_version == 0:
                    fixed_param = _transpose_first_dim(param.data, 3, True, model)
                elif checkpoint_version == 1.0:
                    fixed_param = _transpose_first_dim(param.data, 3, False, model)
                else:
                    print_rank_0(f"Invalid checkpoint version {checkpoint_version}.")
                    sys.exit()
                param.data.copy_(fixed_param)
            if name.endswith(('.key_value.weight', '.key_value.bias')):
                if checkpoint_version == 0:
                    fixed_param = _transpose_first_dim(param.data, 2, True, model)
                elif checkpoint_version == 1.0:
                    fixed_param = _transpose_first_dim(param.data, 2, False, model)
                else:
                    print_rank_0(f"Invalid checkpoint version {checkpoint_version}.")
                    sys.exit()
                param.data.copy_(fixed_param)
        print_rank_0(" succesfully fixed query-key-values ordering for"
                     " checkpoint version {}".format(checkpoint_version))


def _load_base_checkpoint(load_dir, rank0=False, sharded_state_dict=None,
                          exit_on_missing_checkpoint=False, checkpoint_step = None):
    """ Load the base state_dict from the given directory

    If rank0 is true, just loads rank 0 checkpoint, ignoring arguments.
    """
    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_dir)

    # If no tracker file, return nothing
    if not os.path.isfile(tracker_filename):
        if not rank0:
            print_rank_0('WARNING: could not find the metadata file {} '.format(
                tracker_filename))
            print_rank_0('    will not load any checkpoints and will start from '
                         'random')

        # Conditionally exit if checkpoint not found.
        if exit_on_missing_checkpoint:
            print_rank_0(">> '--exit-on-missing-checkpoint' set ... exiting. <<")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            sys.exit()

        return None, "", False

    # Otherwise, read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    if checkpoint_step is not None:
        iteration = checkpoint_step
        release = False
    else:
        iteration, release = read_metadata(tracker_filename)

    # Checkpoint.
    if rank0:
        checkpoint_name = find_checkpoint_rank_0(load_dir, iteration, release)
        is_dist_ckpt = checkpoint_name is not None and dist_checkpointing.check_is_distributed_checkpoint(checkpoint_name)
    else:
        checkpoint_name = get_checkpoint_name(load_dir, iteration, release,
                                              return_base_dir=True)
        is_dist_ckpt = dist_checkpointing.check_is_distributed_checkpoint(checkpoint_name)
        if not is_dist_ckpt:
            checkpoint_name = get_checkpoint_name(load_dir, iteration, release,
                                                  return_base_dir=False)
        dist_infix = "distributed " if is_dist_ckpt else ""
        if release:
            print_rank_0(f' loading release {dist_infix}checkpoint from {load_dir}')
        else:
            print_rank_0(f' loading {dist_infix}checkpoint from {load_dir} at iteration {iteration}')

    # Load the checkpoint.
    if is_dist_ckpt:
        if rank0:
            state_dict = dist_checkpointing.load_common_state_dict(checkpoint_name)
            return state_dict, checkpoint_name, release

        # at this point args are available
        args = get_args()
        if sharded_state_dict is None:
            assert not args.auto_detect_ckpt_format and not args.use_dist_ckpt, (args.auto_detect_ckpt_format, args.use_dist_ckpt)
            raise RuntimeError('Detected load from a distributed checkpoint, but neither --use-dist-ckpt nor --auto-detect-ckpt-format is set.')

        load_strategy = get_default_load_sharded_strategy(checkpoint_name)
        if args.ckpt_fully_parallel_load:
            load_strategy = FullyParallelLoadStrategyWrapper(load_strategy,
                                                             mpu.get_data_parallel_group(with_context_parallel=True))
        state_dict = dist_checkpointing.load(sharded_state_dict, checkpoint_name, load_strategy, strict=args.dist_ckpt_strictness)
        return state_dict, checkpoint_name, release

    try:
        state_dict = torch.load(checkpoint_name, map_location='cpu')
    except ModuleNotFoundError:
        from megatron.legacy.fp16_deprecated import loss_scaler
        # For backward compatibility.
        if not rank0:
            print_rank_0(' > deserializing using the old code structure ...')
        sys.modules['fp16.loss_scaler'] = sys.modules[
            'megatron.legacy.fp16_deprecated.loss_scaler']
        sys.modules['megatron.fp16.loss_scaler'] = sys.modules[
            'megatron.legacy.fp16_deprecated.loss_scaler']
        sys.modules['megatron.model'] = sys.modules['megatron.legacy.model']
        state_dict = torch.load(checkpoint_name, map_location='cpu')
        sys.modules.pop('fp16.loss_scaler', None)
        sys.modules.pop('megatron.fp16.loss_scaler', None)
        sys.modules.pop('megatron.model', None)
    except BaseException as e:
        print('could not load the checkpoint')
        print(e)
        sys.exit()

    return state_dict, checkpoint_name, release


def load_args_from_checkpoint(args, load_arg='load',
                              exit_on_missing_checkpoint=False):
    """Set required arguments from the checkpoint specified in the
    arguments.

    Will overwrite arguments that have a non-None default value, but
    will leave any arguments that default to None as set.

    Returns the same args NameSpace with the new values added/updated.

    If no checkpoint is specified in args, or if the checkpoint is
    there but invalid, the arguments will not be modified

    """
    load_dir = getattr(args, load_arg)

    if load_dir is None:
        print_rank_0('No load directory specified, using provided arguments.')
        return args

    state_dict, checkpoint_name, release = _load_base_checkpoint(
        load_dir,
        rank0=True,
        exit_on_missing_checkpoint=exit_on_missing_checkpoint,
        checkpoint_step=args.ckpt_step
    )

    # Args.
    if not state_dict:
        print_rank_0('Checkpoint not found to provide arguments, using provided arguments.')
        return args

    if 'args' not in state_dict:
        print_rank_0('Checkpoint provided does not have arguments saved, using provided arguments.')
        return args

    checkpoint_args = state_dict['args']
    checkpoint_version = state_dict.get('checkpoint_version', 0)
    args.iteration = state_dict['iteration']

    # One-off conversion for foundation models
    if hasattr(checkpoint_args, 'disable_bias_linear'):
        setattr(checkpoint_args, 'add_bias_linear', not getattr(checkpoint_args, 'disable_bias_linear'))

    def _set_arg(arg_name, old_arg_name=None, force=False):
        if not force and getattr(args, arg_name, None) is not None:
            return

        if old_arg_name is not None:
            checkpoint_value = getattr(checkpoint_args, old_arg_name, None)
        else:
            checkpoint_value = getattr(checkpoint_args, arg_name, None)

        if checkpoint_value is not None:
            print_rank_0(f"Setting {arg_name} to {checkpoint_value} from checkpoint")
            setattr(args, arg_name, checkpoint_value)
        else:
            print_rank_0(f"Checkpoint did not provide arguments {arg_name}")

    _set_arg('num_layers')
    _set_arg('hidden_size')
    _set_arg('ffn_hidden_size')
    _set_arg('seq_length')
    _set_arg('num_attention_heads')
    _set_arg('num_query_groups', force=True)
    _set_arg('group_query_attention', force=True)
    _set_arg('kv_channels')
    _set_arg('max_position_embeddings')
    _set_arg('position_embedding_type', force=True)
    _set_arg('add_position_embedding', force=True)
    _set_arg('use_rotary_position_embeddings', force=True)
    _set_arg('rotary_percent', force=True)
    _set_arg('rotary_interleaved', force=True)
    _set_arg('add_bias_linear', force=True)
    _set_arg('add_qkv_bias', force=True)
    _set_arg('swiglu', force=True)
    _set_arg('untie_embeddings_and_output_weights', force=True)
    _set_arg('apply_layernorm_1p', force=True)
    _set_arg('normalization', force=True)
    _set_arg('tokenizer_type')
    _set_arg('padded_vocab_size')
    _set_arg('apply_query_key_layer_scaling', force=True)
    if checkpoint_version < 3.0:
        _set_arg('tensor_model_parallel_size',
                 'model_parallel_size')
    else:
        _set_arg('tensor_model_parallel_size', force=True)
        _set_arg('pipeline_model_parallel_size', force=True)
        _set_arg('virtual_pipeline_model_parallel_size', force=True)
        _set_arg('num_layers_per_virtual_pipeline_stage')
    return args, checkpoint_args


def load_checkpoint(model, optimizer, opt_param_scheduler, load_arg='load', strict=True):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """
    args = get_args()
    load_dir = getattr(args, load_arg)

    # Finetuning directories
    pretrained_dir = getattr(args,'pretrained_checkpoint', None)
    if pretrained_dir is not None and not checkpoint_exists(load_dir):
        print_rank_0(f'Checkpoint file not found in load directory {load_dir} attempting to finetune with checkpoint in {pretrained_dir}')
        load_dir = pretrained_dir
        if not checkpoint_exists(load_dir):
            raise FileNotFoundError("No checkpoint found in load directory or pretrained directory")
        args.finetune = True


    model = unwrap_model(model)

    load_kwargs = {}
    is_dist_ckpt = False
    if args.auto_detect_ckpt_format or args.use_dist_ckpt:
        state_dict, checkpoint_name, release = _load_base_checkpoint(load_dir, rank0=True, exit_on_missing_checkpoint=args.exit_on_missing_checkpoint)
        is_dist_ckpt = dist_checkpointing.check_is_distributed_checkpoint(checkpoint_name)
        if is_dist_ckpt:
            ckpt_tp_pp = (state_dict['args'].tensor_model_parallel_size, state_dict['args'].pipeline_model_parallel_size)
            run_tp_pp = (mpu.get_tensor_model_parallel_world_size(), mpu.get_pipeline_model_parallel_world_size())
            mismatch_msg = "(TP, PP) mismatch after resume ({} vs {} from checkpoint)".format(ckpt_tp_pp, run_tp_pp)

            # Determine if RNG state will be loaded
            if (ckpt_tp_pp == run_tp_pp and not release and not args.finetune and not args.no_load_rng
                    and not getattr(state_dict['args'], 'no_save_rng', False)):
                gen_sd_rng_state = get_rng_state(True)  # we can load the rng state
            else:
                gen_sd_rng_state = None
                if ckpt_tp_pp != run_tp_pp:
                    print_rank_0("{}: RNG state will be ignored".format(mismatch_msg))

            optim_sd_kwargs = dict(is_loading=True)
            # Determine if optimizer state will be loaded
            if (not release and not args.finetune and not args.no_load_optim
                    and not getattr(state_dict['args'], 'no_save_optim', False)):
                gen_sd_optim = optimizer
                gen_sd_opt_param_scheduler = opt_param_scheduler

                if args.use_distributed_optimizer:
                    optim_sd_kwargs['sharding_type'] = ('fully_sharded_model_space'
                                                        if getattr(state_dict['args'], 'ckpt_fully_parallel_save', False)
                                                        else 'dp_zero_gather_scatter')
                    # This is for backwards-compatibility. Can be removed once 'fully_sharded_bucket_space' loading is removed
                    for maybe_dist_opt_optim_state in (state_dict['optimizer'], *state_dict['optimizer'].values()):
                        if 'param_state_sharding_type' in maybe_dist_opt_optim_state:
                            if maybe_dist_opt_optim_state['param_state_sharding_type'] == 'fully_sharded_bucket_space':
                                print_rank_0('Detected deprecated `fully_sharded_bucket_space` DistributedOptimizer checkpoint format')
                                optim_sd_kwargs['sharding_type'] = maybe_dist_opt_optim_state['param_state_sharding_type']
                            break

                    if ckpt_tp_pp != run_tp_pp and optim_sd_kwargs['sharding_type'] != 'fully_sharded_model_space':
                        raise RuntimeError(f"{mismatch_msg}: not supported for DistributedOptimizer with sharding type {optim_sd_kwargs['sharding_type']}."
                                           f" Please use `--ckpt-fully-parallel-save` flag during checkpoint saving.")
            else:
                gen_sd_optim = None
                gen_sd_opt_param_scheduler = None
            load_kwargs['sharded_state_dict'] = generate_state_dict(args, model, gen_sd_optim, gen_sd_opt_param_scheduler,
                                                                    gen_sd_rng_state, True, optim_sd_kwargs=optim_sd_kwargs)
            load_kwargs['exit_on_missing_checkpoint'] = args.exit_on_missing_checkpoint

    state_dict, checkpoint_name, release = _load_base_checkpoint(load_dir, rank0=False, **load_kwargs)

    # Checkpoint not loaded.
    if state_dict is None:
        # Iteration and num_floating_point_operations_so_far default to 0.
        return 0, 0

    # Set checkpoint version.
    set_checkpoint_version(state_dict.get('checkpoint_version', 0))

    # Set iteration.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = state_dict['iteration']
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = state_dict['total_iters']
            except KeyError:
                print_rank_0('A metadata file exists but unable to load '
                             'iteration from checkpoint {}, exiting'.format(checkpoint_name))
                sys.exit()
    num_floating_point_operations_so_far = state_dict.get('num_floating_point_operations_so_far', 0)

    # Check arguments.
    assert args.consumed_train_samples == 0
    assert args.consumed_valid_samples == 0
    if 'args' in state_dict and not args.finetune:
        checkpoint_args = state_dict['args']
        check_checkpoint_args(checkpoint_args)
        args.consumed_train_samples = getattr(checkpoint_args,
                                              'consumed_train_samples', 0)
        update_num_microbatches(consumed_samples=args.consumed_train_samples)
        args.consumed_valid_samples = getattr(checkpoint_args,
                                              'consumed_valid_samples', 0)
    else:
        print_rank_0('could not find arguments in the checkpoint ...')

    # [ModelOpt]: loading modelopt_state (sharded or not)
    if has_nvidia_modelopt:
        if args.use_dist_ckpt:
            restore_sharded_modelopt_state(model, checkpoint_name)
        else:
            restore_modelopt_state(model, state_dict)

    # Model.
    strict = False if args.retro_add_retriever else strict
    if len(model) == 1:
        model[0].load_state_dict(state_dict['model'], strict=strict)
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            model[i].load_state_dict(state_dict['model%d' % i], strict=strict)

    # Fix up query/key/value matrix ordering if needed.
    checkpoint_version = get_checkpoint_version()
    print_rank_0(f' checkpoint version {checkpoint_version}')
    fix_query_key_value_ordering(model, checkpoint_version)

    # Optimizer.
    if not release and not args.finetune and not args.no_load_optim:
        try:
            # Load state dict.
            if optimizer is not None:
                optimizer.load_state_dict(state_dict['optimizer'])

            # Load distributed optimizer's custom parameter state.
            # For distributed checkpoint it's already loaded in load_state_dict above
            if args.use_distributed_optimizer and not is_dist_ckpt:
                tracker_filename = get_checkpoint_tracker_filename(load_dir)
                iteration, release = read_metadata(tracker_filename)
                model_checkpoint_name = \
                    get_checkpoint_name(load_dir, iteration, release)
                optim_checkpoint_name = \
                    get_distributed_optimizer_checkpoint_name(
                        model_checkpoint_name)
                optimizer.load_parameter_state(optim_checkpoint_name)

            # Load scheduler.
            if opt_param_scheduler is not None:
                if 'lr_scheduler' in state_dict: # backward compatbility
                    opt_param_scheduler.load_state_dict(state_dict['lr_scheduler'])
                else:
                    opt_param_scheduler.load_state_dict(state_dict['opt_param_scheduler'])
        except KeyError:
            print_rank_0('Unable to load optimizer from checkpoint {}. '
                         'Specify --no-load-optim or --finetune to prevent '
                         'attempting to load the optimizer state, '
                         'exiting ...'.format(checkpoint_name))
            sys.exit()
    else:
        if (args.fp16 or args.bf16) and optimizer is not None:
            optimizer.reload_model_params()

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            if 'rng_state' in state_dict:
                # access rng_state for data parallel rank
                if args.data_parallel_random_init:
                    rng_state = state_dict['rng_state'][mpu.get_data_parallel_rank()]
                else:
                    rng_state = state_dict['rng_state'][0]
                random.setstate(rng_state['random_rng_state'])
                np.random.set_state(rng_state['np_rng_state'])
                torch.set_rng_state(rng_state['torch_rng_state'])
                torch.cuda.set_rng_state(rng_state['cuda_rng_state'])
                # Check for empty states array
                if not rng_state['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    rng_state['rng_tracker_states'])
            else:  # backward compatability
                random.setstate(state_dict['random_rng_state'])
                np.random.set_state(state_dict['np_rng_state'])
                torch.set_rng_state(state_dict['torch_rng_state'])
                torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
                # Check for empty states array
                if not state_dict['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    state_dict['rng_tracker_states'])
        except KeyError:
            print_rank_0('Unable to load rng state from checkpoint {}. '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the rng state, '
                         'exiting ...'.format(checkpoint_name))
            sys.exit()

    # Some utilities want to load a checkpoint without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(f'  successfully loaded checkpoint from {load_dir} '
                 f'[ t {mpu.get_tensor_model_parallel_rank()}, '
                 f'p {mpu.get_pipeline_model_parallel_rank()} ] '
                 f'at iteration {iteration}')

    return iteration, num_floating_point_operations_so_far


def load_biencoder_checkpoint(model, only_query_model=False,
                              only_context_model=False, custom_load_path=None):
    """
    selectively load retrieval models for indexing/retrieving
    from saved checkpoints
    """

    args = get_args()

    model = unwrap_model(model)

    load_path = custom_load_path if custom_load_path is not None else args.load

    tracker_filename = get_checkpoint_tracker_filename(load_path)
    with open(tracker_filename, 'r') as f:
        iteration = int(f.read().strip())

    checkpoint_name = get_checkpoint_name(load_path, iteration,
                                          args.use_distributed_optimizer,
                                          release=False)

    if mpu.get_data_parallel_rank() == 0:
        print('global rank {} is loading checkpoint {}'.format(
            torch.distributed.get_rank(), checkpoint_name))

    state_dict = torch.load(checkpoint_name, map_location='cpu')
    ret_state_dict = state_dict['model']

    if only_query_model:
        ret_state_dict.pop('context_model')
    if only_context_model:
        ret_state_dict.pop('query_model')

    assert len(model) == 1
    model[0].load_state_dict(ret_state_dict)
    torch.distributed.barrier()

    if mpu.get_data_parallel_rank() == 0:
        print(' successfully loaded {}'.format(checkpoint_name))

    return model
