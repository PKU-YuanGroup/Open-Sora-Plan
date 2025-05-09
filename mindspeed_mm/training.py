# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import os
import gc
import sys
import time
import torch
import torch_npu

from megatron.core.optimizer.clip_grads import AdaptiveGradClipInfo

from megatron.core import mpu
from megatron.core.utils import get_model_config
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.training.checkpointing import save_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training.initialize import set_jit_fusion_options
from megatron.training.initialize import write_args_to_tensorboard
from megatron.training.global_vars import (
    get_args,
    get_signal_handler,
    get_timers,
    get_tensorboard_writer,
    get_wandb_writer,
    get_one_logger,
)
from megatron.core.num_microbatches_calculator import (
    get_num_microbatches,
    update_num_microbatches,
)
from megatron.training.training import (
    get_num_microbatches,
    training_log,
    evaluate_and_print_results,
    save_checkpoint_and_time,
    print_datetime,
    num_floating_point_operations,
    get_one_logger,
    append_to_progress_log,
    build_train_valid_test_data_iterators,
    setup_model_and_optimizer,
)
from megatron.training.utils import (
    calc_params_l2_norm,
    check_adlr_autoresume_termination,
    print_rank_0,
    unwrap_model,
)

from mindspeed_mm.configs.config import merge_mm_args
from mindspeed_mm.tools.profiler import Profiler

_TRAIN_START_TIME = time.time()


def pretrain(
    data_meta_info_list,
    train_valid_test_dataset_provider,
    model_provider,
    model_type,
    forward_step_func,
    process_non_loss_data_func=None,
    extra_args_provider=None,
    args_defaults={},
):
    """
    Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Args:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(
        extra_args_provider=extra_args_provider, args_defaults=args_defaults
    )

    torch.distributed.barrier()

    init_func = args_defaults.get("init_func", None)
    if init_func:
        init_func()

    args = get_args()
    merge_mm_args(args)
    timers = get_timers()
    if args.log_progress:
        append_to_progress_log("Starting job")

    torch.backends.cuda.matmul.allow_tf32 = getattr(args.mm.model, "allow_tf32", False)
    torch.npu.config.allow_internal_format = getattr(args.mm.model, "allow_internal_format", False)

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor(
        [_TRAIN_START_TIME], dtype=torch.float, device="cuda"
    )
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0(
        "time to initialize megatron (seconds): {:.3f}".format(
            time.time() - _TRAIN_START_TIME
        )
    )
    print_datetime("after megatron is initialized")

    args = get_args()
    if args.save_interval == 0 or args.log_interval == 0 or args.eval_interval == 0:
        raise ValueError("save_interval, log_interval, and eval_interval cannot be 0")
    timers = get_timers()

    one_logger = get_one_logger()
    if one_logger:
        one_logger.log_metrics({"train_iterations_warmup": 5})

    torch.distributed.barrier()
    # Model, optimizer, and learning rate.
    timers("model-and-optimizer-setup", log_level=0).start(barrier=True)
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type
    )
    timers("model-and-optimizer-setup").stop()
    torch.distributed.barrier()
    print_datetime("after model, optimizer, and learning rate scheduler are built")
    config = get_model_config(model[0])

    dataset_index = 0
    dataset_index_txt = os.path.join(args.save, 'dataset_index.txt')
    if os.path.exists(dataset_index_txt):
        with open(dataset_index_txt, 'r') as f:
            dataset_index = int(f.read().strip())

    extreme_error_flag = False

    if dataset_index < len(data_meta_info_list):
        print_rank_0(f"dataset_index: {dataset_index}")
        data_meta_info = data_meta_info_list[dataset_index]
        if not os.path.exists(data_meta_info) or not data_meta_info.endswith('.txt'):
            raise ValueError(f"data_meta_info: {data_meta_info} is not a valid file path")
        with open(data_meta_info, 'r') as f:
            info = f.read().strip()
        print_rank_0(f"data_meta_info: {data_meta_info}, content: {info}")
        setattr(args.mm.data.dataset_param.basic_parameters, 'data_path', data_meta_info)
        torch.distributed.barrier()
        # NOTE For group data, we need to set initial_global_step_for_sampler
        group_data = getattr(args.mm.data.dataloader_param, 'group_data', False)
        if group_data:
            # group sampler
            timers("global-step-for-sampler-setup", log_level=0).start()
            print_rank_0("use group sampler...")
            global_step_for_sampler_txt = os.path.join(args.save, 'global_step_for_sampler.txt')
            if os.path.exists(global_step_for_sampler_txt):
                with open(global_step_for_sampler_txt, 'r') as f:
                    global_step_for_sampler = int(f.read().strip())
            elif args.mm.data.dataloader_param.initial_global_step_for_sampler != 0:
                global_step_for_sampler = args.mm.data.dataloader_param.initial_global_step_for_sampler
            else:
                global_step_for_sampler = 0
            setattr(args.mm.data.dataloader_param, 'initial_global_step_for_sampler', global_step_for_sampler)
            print_rank_0(f"global_step_for_sampler: {args.mm.data.dataloader_param.initial_global_step_for_sampler}")
            timers("global-step-for-sampler-setup").stop()
        torch.distributed.barrier()
        # Data stuff.
        timers("train/valid/test-data-iterators-setup", log_level=0).start(barrier=True)
        if args.virtual_pipeline_model_parallel_size is not None:
            train_data_iterator = []
            valid_data_iterator = []
            test_data_iterator = []
            for i in range(len(model)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                iterators = build_train_valid_test_data_iterators(
                    train_valid_test_dataset_provider
                )
                train_data_iterator.append(iterators[0])
                valid_data_iterator.append(iterators[1])
                test_data_iterator.append(iterators[2])
        else:
            train_data_iterator, valid_data_iterator, test_data_iterator = (
                build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
            )
        timers("train/valid/test-data-iterators-setup").stop()
        print_datetime("after dataloaders are built")

        # Print setup timing.
        print_rank_0("done with setup ...")
        timers.log(
            ["model-and-optimizer-setup", "train/valid/test-data-iterators-setup"],
            barrier=True,
        )
        torch.distributed.barrier()
        extreme_error_flag = train_on_one_dataset(
            forward_step_func,
            model,
            optimizer,
            opt_param_scheduler,
            train_data_iterator,
            valid_data_iterator,
            process_non_loss_data_func,
            config,
        )
        if not extreme_error_flag:
            print_rank_0(f'dataset_index: {dataset_index}, done with training')
            print(f"rank = {torch.distributed.get_rank()}, done with training")
            dataset_index += 1
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                print("update dataset_index.txt")
                with open(dataset_index_txt, 'w') as f:
                    f.write(str(dataset_index))
        else:
            print_rank_0("extreme error, we must restart training on this dataset")

        torch.distributed.barrier()

        if dataset_index < len(data_meta_info_list):
            raise ValueError("In RDZV training mode, we actively throw exceptions to execute the next task")
        else:
            print_rank_0("training is done, exit")
        
    print(f"rank = {torch.distributed.get_rank()}, done with training all datasets")
    if torch.distributed.get_rank() == 0:
        print("training is done, remove dataset_index.txt")
        os.remove(dataset_index_txt)
    torch.distributed.barrier()


def train_on_one_dataset(
    forward_step_func,
    model,
    optimizer,
    opt_param_scheduler,
    train_data_iterator,
    valid_data_iterator,
    process_non_loss_data_func,
    config,
):

    args = get_args()
    timers = get_timers()

    extreme_error_flag = False
    if not args.skip_train:
        print_rank_0("training ...")

        if args.dataloader_type == "cyclic" and args.retro_project_dir:
            if args.retro_cyclic_train_iters is None:
                raise AssertionError
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        torch.distributed.barrier()        
        iteration = 0
        if args.do_train and args.train_iters > 0:
            iteration, num_floating_point_operations_so_far, extreme_error_flag = train(
                forward_step_func,
                model,
                optimizer,
                opt_param_scheduler,
                train_data_iterator,
                valid_data_iterator,
                process_non_loss_data_func,
                config,
            )
        torch.distributed.barrier()
        print_datetime("after training is done")

        print(f'rank = {torch.distributed.get_rank()}, before save last checkpoint')
        if args.save and iteration != 0 and iteration % args.save_interval != 0:
            print_rank_0(f'Training ends, save checkpoint at iteration {iteration}')
            save_checkpoint(
                iteration,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
            )
        torch.distributed.barrier()
        global_step_for_sampler_txt = os.path.join(args.save, 'global_step_for_sampler.txt')
        print(f'rank = {torch.distributed.get_rank()}, before delete global_step_for_sampler_txt')
        if torch.distributed.get_rank() == 0:
            # NOTE For group data, we need to save the global step for sampler.
            group_data = getattr(args.mm.data.dataloader_param, 'group_data', False)
            if group_data and not extreme_error_flag:
                # group sampler
                timers("global-step-for-sampler-txt-delete-setup", log_level=0).start()
                if os.path.exists(global_step_for_sampler_txt):
                    print_rank_0("delete global_step_for_sampler.txt...")
                    with open(global_step_for_sampler_txt, 'r') as f:
                        global_step_for_sampler = int(f.read().strip())
                    print_rank_0(f"global_step_for_sampler: {global_step_for_sampler}, and we will reset it to 0...")
                    os.remove(global_step_for_sampler_txt)
                timers("global-step-for-sampler-txt-delete-setup").stop()
        print(f'rank = {torch.distributed.get_rank()}, after delete global_step_for_sampler_txt')
        torch.distributed.barrier()
        if not os.path.exists(global_step_for_sampler_txt):
            print("reset global_step_for_sampler to 0, global_step_for_sampler_txt is deleted")
        else:
            raise Exception("error! global_step_for_sampler_txt is not deleted")
    else:
        print_rank_0("skipping training (--skip-train is on) ...")

        iteration = args.iteration

    if args.do_valid:
        prefix = f"iteration {iteration} on validation set"
        evaluate_and_print_results(
            prefix,
            forward_step_func,
            valid_data_iterator,
            model,
            iteration,
            process_non_loss_data_func,
            config,
            verbose=True,
            write_to_tensorboard=not args.skip_train,
        )

    if args.do_test:
        prefix = f"iteration {iteration} on test set"
        evaluate_and_print_results(
            prefix,
            forward_step_func,
            test_data_iterator,
            model,
            iteration,
            process_non_loss_data_func,
            config,
            verbose=True,
            write_to_tensorboard=not args.skip_train,
        )

    return extreme_error_flag

def train(
    forward_step_func,
    model,
    optimizer,
    opt_param_scheduler,
    train_data_iterator,
    valid_data_iterator,
    process_non_loss_data_func,
    config,
):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    # Write args to tensorboard
    write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration
    # NOTE for group data, we need to log initial iteration for computing global_step_for_sampler
    initial_iteration = iteration

    one_logger = get_one_logger()
    if one_logger:
        iteration_start = iteration
        train_samples_start = args.consumed_train_samples
        train_samples_target = args.train_samples
        one_logger.log_metrics(
            {
                "train_samples_start": args.consumed_train_samples,
                "train_iterations_start": iteration,
                "train_samples_target": train_samples_target,
                "train_iterations_target": args.train_iters,
            }
        )

    num_floating_point_operations_so_far = args.num_floating_point_operations_so_far

    # Setup some training config params
    config.grad_scale_func = optimizer.scale_loss
    config.timers = timers
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        if config.no_sync_func is not None:
            raise AssertionError(
                "When overlap_grad_reduce is True, config.no_sync_func must be None; "
                "a custom no_sync_func is not supported when overlapping grad-reduce"
            )
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.delay_grad_reduce:
            config.grad_sync_func = [
                model_chunk.start_grad_sync for model_chunk in model
            ]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.delay_param_gather:
        config.param_sync_func = [
            lambda x, model_index=model_index: optimizer.finish_param_sync(
                model_index, x
            )
            for model_index in range(len(model))
        ]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads

    timers("interval-time", log_level=0).start(barrier=True)
    print_datetime("before the start of training step")
    report_memory_flag = True
    exit_flag = False

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        if args.manual_gc_interval < 0:
            raise AssertionError(
                "Manual garbage collection interval should be larger than or equal to 0."
            )
        gc.disable()
        gc.collect()

    num_microbatches = get_num_microbatches()
    eval_duration = 0.0
    eval_iterations = 0

    def track_e2e_metrics():
        # Nested function to track a bunch of E2E APP metrics
        if one_logger:
            # overall_elapsed
            train_duration = timers("interval-time").active_time()
            train_samples = args.consumed_train_samples - train_samples_start
            train_iterations = iteration - iteration_start
            train_iterations_time_msecs_avg = (
                (train_duration * 1000.0) / train_iterations
                if train_iterations > 0
                else None
            )
            if eval_iterations > 0:
                validation_iterations_time_msecs_avg = (
                    eval_duration * 1000.0
                ) / eval_iterations
            else:
                validation_iterations_time_msecs_avg = None

            one_logger.log_metrics(
                {
                    "train_iterations_end": iteration,
                    "train_samples_end": args.consumed_train_samples,
                    "train_iterations": train_iterations,
                    "train_samples": train_samples,
                    "train_iterations_time_msecs_avg": train_iterations_time_msecs_avg,
                    "validation_iterations_time_msecs_avg": validation_iterations_time_msecs_avg,
                }
            )

    prof = Profiler(args.mm.tool.profile)
    prof.start()

    while iteration < args.train_iters:
        # Update number of microbatches first without consistency check to decide if a
        # checkpoint should be saved. If the number of microbatches is different
        # from the previous iteration, save a checkpoint. Then run consistency check
        # to make sure training configuration is still valid.
        update_num_microbatches(args.consumed_train_samples, consistency_check=False)
        if get_num_microbatches() != num_microbatches and iteration != 0:
            if get_num_microbatches() <= num_microbatches:
                raise AssertionError(
                    "number of microbatches should be increasing due to batch size rampup"
                )
            save_checkpoint_and_time(
                iteration,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                None,
            )
        num_microbatches = get_num_microbatches()
        update_num_microbatches(args.consumed_train_samples, consistency_check=True)

        args.curr_iteration = iteration

        data_run_out = False
        try:
            loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = train_step(
                forward_step_func,
                train_data_iterator,
                model,
                optimizer,
                opt_param_scheduler,
                config,
            )
        except StopIteration:
            data_run_out = True
            print("Training is done because dataloader run out of data.")
        except Exception as e:
            print(f"Training is done because of exception {type(e).__name__}")
            raise e
    
        iteration += 1
        batch_size = (
            mpu.get_data_parallel_world_size()
            * args.micro_batch_size
            * get_num_microbatches()
        )
        args.consumed_train_samples += batch_size
        num_floating_point_operations_so_far += num_floating_point_operations(
            args, batch_size
        )

        # Logging.
        loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)

        if iteration % args.log_interval == 0:
            track_e2e_metrics()

        learning_rate = None
        decoupled_learning_rate = None
        for param_group in optimizer.param_groups:
            if param_group["is_decoupled_lr"]:
                decoupled_learning_rate = param_group["lr"]
            else:
                learning_rate = param_group["lr"]

        extreme_error_flag = False
        if AdaptiveGradClipInfo.extreme_error_flag:
            print_rank_0("Extreme error, stop training!")
            extreme_error_flag = True
            
        report_memory_flag = training_log(
            loss_dict,
            total_loss_dict,
            learning_rate,
            decoupled_learning_rate,
            iteration,
            loss_scale,
            report_memory_flag,
            skipped_iter,
            grad_norm,
            params_norm,
            num_zeros_in_grad,
        )
        if torch.distributed.get_rank() == 0:
            if extreme_error_flag: 
                # NOTE For group data, we need to save the global step for sampler.
                group_data = getattr(args.mm.data.dataloader_param, 'group_data', False)
                if group_data:
                    # group sampler
                    timers("global-step-for-sampler-txt-save-setup", log_level=0).start()
                    print_rank_0("save global_step_for_sampler.txt")
                    initial_global_step_for_sampler = getattr(args.mm.data.dataloader_param, 'initial_global_step_for_sampler', 0)
                    global_step_for_sampler_txt = os.path.join(args.save, 'global_step_for_sampler.txt')
                    os.makedirs(os.path.dirname(global_step_for_sampler_txt), exist_ok=True)
                    global_step_for_sampler = initial_global_step_for_sampler + iteration - initial_iteration
                    print_rank_0(f"initial_global_step_for_sampler: {initial_global_step_for_sampler}")
                    print_rank_0(f"current_global_step_for_sampler: {global_step_for_sampler}")
                    with open(global_step_for_sampler_txt, 'w') as f:
                        f.write(str(global_step_for_sampler))
                    timers("global-step-for-sampler-txt-save-setup").stop()
        torch.distributed.barrier()
        # Autoresume
        if args.adlr_autoresume and (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(
                iteration, model, optimizer, opt_param_scheduler
            )

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and args.do_valid:
            timers("interval-time").stop()
            if args.use_distributed_optimizer and args.overlap_param_gather:
                optimizer.disable_pre_hook()
            if args.manual_gc and args.manual_gc_eval:
                # Collect all objects.
                gc.collect()
            prefix = "iteration {}".format(iteration)
            timers("eval-time", log_level=0).start(barrier=True)
            evaluate_and_print_results(
                prefix,
                forward_step_func,
                valid_data_iterator,
                model,
                iteration,
                process_non_loss_data_func,
                config,
                False,
            )
            eval_duration += timers("eval-time").elapsed()
            eval_iterations += args.eval_iters
            timers("eval-time").stop()
            if args.manual_gc and args.manual_gc_eval:
                # Collect only the objects created and used in evaluation.
                gc.collect(generation=0)
            if args.use_distributed_optimizer and args.overlap_param_gather:
                optimizer.enable_pre_hook()
            timers("interval-time", log_level=0).start(barrier=True)

        # Checkpointing
        saved_checkpoint = False
        if args.exit_signal_handler:
            signal_handler = get_signal_handler()
            if any(signal_handler.signals_received()):
                save_checkpoint_and_time(
                    iteration,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far,
                    None,
                )
                print_datetime("exiting program after receiving SIGTERM.")
                exit_flag = True
                break

        if args.save and args.save_interval and iteration % args.save_interval == 0:
            save_checkpoint_and_time(
                iteration,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                None,
            )
            saved_checkpoint = True

            if torch.distributed.get_rank() == 0:
                # NOTE For group data, we need to save the global step for sampler.
                group_data = getattr(args.mm.data.dataloader_param, 'group_data', False)
                if group_data:
                    # group sampler
                    timers("global-step-for-sampler-txt-save-setup", log_level=0).start()
                    print_rank_0("save global_step_for_sampler.txt")
                    initial_global_step_for_sampler = getattr(args.mm.data.dataloader_param, 'initial_global_step_for_sampler', 0)
                    global_step_for_sampler_txt = os.path.join(args.save, 'global_step_for_sampler.txt')
                    global_step_for_sampler = initial_global_step_for_sampler + iteration - initial_iteration
                    print_rank_0(f"initial_global_step_for_sampler: {initial_global_step_for_sampler}")
                    print_rank_0(f"current_global_step_for_sampler: {global_step_for_sampler}")
                    os.makedirs(os.path.dirname(global_step_for_sampler_txt), exist_ok=True)
                    with open(global_step_for_sampler_txt, 'w') as f:
                        f.write(str(global_step_for_sampler))
                    timers("global-step-for-sampler-txt-save-setup").stop()
            torch.distributed.barrier()

        # Exiting based on duration
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.tensor(
                [train_time > args.exit_duration_in_mins],
                dtype=torch.int,
                device="cuda",
            )
            torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(
                        iteration,
                        model,
                        optimizer,
                        opt_param_scheduler,
                        num_floating_point_operations_so_far,
                        None,
                    )
                print_datetime("exiting program after {} minutes".format(train_time))
                exit_flag = True
                break

        # Exiting based on iterations
        if args.exit_interval and iteration % args.exit_interval == 0:
            if args.save and not saved_checkpoint:
                save_checkpoint_and_time(
                    iteration,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far,
                    None,
                )
            torch.distributed.barrier()
            print_datetime("exiting program at iteration {}".format(iteration))
            exit_flag = True
            break

        if args.manual_gc:
            if args.manual_gc_interval != 0 and iteration % args.manual_gc_interval == 0:
                gc.collect()

        prof.step()

        # Exit if data iterator is exhausted or extreme error is detected.
        if data_run_out or extreme_error_flag:
            break

    prof.stop()

    track_e2e_metrics()

    # Flush TensorBoard and WandB writers.
    writer = get_tensorboard_writer()
    if writer:
        writer.flush()
    wandb_writer = get_wandb_writer()
    if wandb_writer:
        wandb_writer.finish()

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if args.use_distributed_optimizer and args.overlap_param_gather:
        optimizer.disable_pre_hook()

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if exit_flag:
        sys.exit()

    return iteration, num_floating_point_operations_so_far, extreme_error_flag


def train_step(
    forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config
):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Set grad to zero.
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    # Forward pass.
    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=get_num_microbatches(),
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False,
    )

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Vision gradients.
    if (
        getattr(args, "vision_pretraining", False)
        and args.vision_pretraining_type == "dino"
    ):
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    timers("optimizer", log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers("optimizer").stop()

    # Vision momentum.
    if (
        getattr(args, "vision_pretraining", False)
        and args.vision_pretraining_type == "dino"
    ):
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:
        increment = (
            get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
        )
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(
                losses_reduced_for_key
            )
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad
