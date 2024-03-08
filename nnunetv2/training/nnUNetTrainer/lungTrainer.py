import medpy.metric
import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.polyp_testloader import test_dataset
# from nnunetv2.training.network.model.dim2.res2unet.res2unet import Res2UNet
from nnunetv2.paths import nnUNet_raw
from einops import *
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import *
# from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer, maybe_mkdir_p, \
#     dist, nnUNetDataset, check_workers_alive_and_busy, time,

from nnunetv2.utilities.lung_get_network_from_plans import get_network_from_plans
from nnunetv2.training.lr_scheduler.polylr import *
import re
# import matplotlib
# matplotlib.use('tkAgg')

# import matplotlib.pyplot as plt


class LungTrainer(nnUNetTrainer):
    base_ch = 16
    block = "FusedMBConv"  # ConvNeXtBlock  FusedMBConv
    use_my_unet = True
    network_name = "my_unet"
    project_prefix = "Brain"
    setting = 2
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), debug=True, job_id=None):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, debug, job_id)
        self.initial_lr = 1e-2

        self.best_score = {"dsc_v": (-1, -1.0, -1.0), "dsc": (-1, -1.0, -1.0)}

        if self.debug:
            self.batch_size = 2
        else:
            pass
            # self.batch_size = 12
        # self.grad_scaler = None
    def _get_deep_supervision_scales(self):
        pool_op_kernel_sizes = self.configuration_manager.pool_op_kernel_sizes
        # pool_op_kernel_sizes = pool_op_kernel_sizes[:-1]
        deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
            pool_op_kernel_sizes), axis=0))
        # deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
        #     self.configuration_manager.pool_op_kernel_sizes), axis=0))[:-2]

        # deep_supervision_scales = deep_supervision_scales[:4]

        return deep_supervision_scales

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {},
                                  weight_ce=1, weight_dice=1.5,
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        # comment by me
        # weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights_IXI so that they sum to 1
        weights = weights / weights.sum()

        # weights = np.array([1] * len(deep_supervision_scales))
        print(f"ds wegihts: {weights}")
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True
                                   ) -> nn.Module:

        return get_network_from_plans(plans_manager, dataset_json, configuration_manager,
                                      num_input_channels, deep_supervision=enable_deep_supervision,
                                      base_ch=LungTrainer.base_ch,
                                      block=LungTrainer.block,
                                      use_my_unet=LungTrainer.use_my_unet,
                                      setting=LungTrainer.setting)

    def configure_optimizers(self):
        params = self.network.parameters()
        # backbone_params = list(map(id, self.network.backbone.parameters()))
        # other_params = filter(lambda p: id(p) not in backbone_params, self.network.parameters())

        # optimizer = torch.optim.SGD(
        #     [
        #         {'params': self.network.backbone.parameters(), 'lr': 1e-5},
        #         {'params': other_params}
        #     ],
        #     self.initial_lr, weight_decay=self.weight_decay,
        #     momentum=0.99, nesterov=True)

        optimizer = torch.optim.SGD(
            [
                # {'params': self.network.backbone.parameters(), 'lr': 1e-5},
                {'params': params}
            ],
            self.initial_lr, weight_decay=self.weight_decay,
            momentum=0.99, nesterov=True)

        lr_scheduler = RetinaPolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        # lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def run_training(self, dataset_id):
        self.on_train_start()
        # print(self.network.__class__.__name__)
        if not self.debug and self.local_rank == 0:
            wandb.login(key="66b58ac7004a123a43487d7a6cf34ebb4571a7ea")
            self.initialize_wandb(project=f"{self.project_prefix}_{self.fold}",
                                  name=f"{self.network.__class__.__name__}_{self.job_id}_"
                                       f"{self.fold}_lr_{self.initial_lr}",
                                  dir=self.output_folder,
                                  id=None)
            print(f"debug: {self.debug}".center(50, "="))

        # self.num_iterations_per_epoch = 2
        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            self.print_to_log_file(f"start training, {self.num_iterations_per_epoch}")
            self.print_to_log_file(f"learning rate: {self.optimizer.param_groups[0]['lr']}")

            if self.debug:
                for batch_id in range(1):
                    # print(f"batch_id: {batch_id}")
                    train_outputs.append(self.train_step(next(self.dataloader_train)))
            else:
                print(f"num of epochs: {self.num_iterations_per_epoch}".center(50, "="))
                for batch_id in range(self.num_iterations_per_epoch):
                    # print(f"batch_id: {batch_id} / {250}===================\r", end="")
                    train_outputs.append(self.train_step(next(self.dataloader_train)))

            self.print_to_log_file(f"finished training")

            self.on_train_epoch_end(train_outputs)

            self.real_validation_brain()

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
            torch.cuda.empty_cache()

        self.on_train_end()

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        # data_1 = rearrange(data, 'N C H W -> N H W C')
        # for i in range(data_1.shape[0]):
        #     plt.imshow(data_1[i]); plt.show()
        target = batch['target']
        # for t in target:
        #     print(np.unique(t.detach().cpu().numpy()))
        data = data.to(torch.float16).to(self.device, non_blocking=True)

        if isinstance(target, list):
            target = [i.to(torch.float16).to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(torch.float16).to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # print(f"data.shape: {data.shape}")
            output = self.network(data)
            # del data
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}
    def real_validation_brain(self):
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                                    use_mirroring=True,
                                    perform_everything_on_gpu=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)

        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = segmentation_export_pool._pool
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            tr_keys, val_keys = self.do_split()
            if self.is_ddp:
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]
                tr_keys = tr_keys[self.local_rank:: dist.get_world_size()]

            dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        num_images_properties_loading_threshold=0)

            dataset_all = nnUNetDataset(self.preprocessed_dataset_folder, tr_keys + val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        num_images_properties_loading_threshold=0)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []
            # for k in list(self.dataset_tr.keys()) + list(dataset_val.keys()):
            keys = list(dataset_val.keys())
            if self.debug:
                keys = keys[:1]

            for k in keys:
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                           allowed_num_queued=2 * len(segmentation_export_pool._pool))
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                               allowed_num_queued=2 * len(
                                                                   segmentation_export_pool._pool))

                # data, seg, properties = dataset_val.load_case(k)
                data, seg, properties = dataset_all.load_case(k)

                if self.is_cascaded:
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                output_filename_truncated = join(validation_output_folder, k)

                start_time = end_time = -1

                # prediction = predictor.predict_sliding_window_return_logits(data)
                try:
                    start_time = time()
                    prediction = predictor.predict_sliding_window_return_logits(data)
                    end_time = time()
                except RuntimeError:
                    predictor.perform_everything_on_gpu = False
                    prediction = predictor.predict_sliding_window_return_logits(data)
                    predictor.perform_everything_on_gpu = True

                assert start_time != -1 and end_time != -1 and start_time < end_time
                self.print_to_log_file(f"predicting {k} took {(end_time-start_time):.2f} s")

                prediction = prediction.cpu()

                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, True),
                        )
                    )
                )
                # for debug purposes
                # export_prediction(prediction_for_export, properties, self.configuration, self.plans, self.dataset_json,
                #              output_filename_truncated, save_probabilities)

                # if needed, export the softmax prediction for the next stage
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)

                        try:
                            # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
                            tmp = nnUNetDataset(expected_preprocessed_folder, [k],
                                                num_images_properties_loading_threshold=0)
                            d, s, p = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!")
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file = join(output_folder, k + '.npz')

                        # resample_and_save(prediction, target_shape, output_file, self.plans_manager, self.configuration_manager, properties,
                        #                   self.dataset_json)
                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction, target_shape, output_file, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json),
                            )
                        ))

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        output_file = join(validation_output_folder, 'summary.json')
        if output_file is not None:
            assert output_file.endswith('.json'), 'output_file should end with .json'

        folder_ref = join(self.preprocessed_dataset_folder_base, 'gt_segmentations')
        folder_pred = validation_output_folder

        files_ref = subfiles(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                              suffix=self.dataset_json["file_ending"], join=False)
        files_pred = subfiles(validation_output_folder,
                              suffix=self.dataset_json["file_ending"], join=False)

        chill = True
        if not chill:
            present = [isfile(join(folder_pred, i)) for i in files_ref]
            assert all(present), "Not all files in folder_pred exist in folder_ref"

        files_ref = [join(folder_ref, os.path.basename(i)) for i in files_pred]
        files_pred = [join(folder_pred, i) for i in files_pred]

        all_dice_vol_level = []
        all_haus95 = []

        def dice_score_vol(y_pred, y_true, num_classes):
            def dc(result, reference):
                result = np.atleast_1d(result.astype(np.bool))
                reference = np.atleast_1d(reference.astype(np.bool))

                intersection = np.count_nonzero(result & reference)

                size_i1 = np.count_nonzero(result)
                size_i2 = np.count_nonzero(reference)

                try:
                    dc = 2. * intersection / float(size_i1 + size_i2)
                except ZeroDivisionError:
                    dc = 1.0

                return dc

            dsc = dc(y_pred, y_true)
            try:
                hd95 = medpy.metric.hd95(y_pred, y_true, voxelspacing=(1.24, 0.79, 0.79))
            except:
                print(f"================ found problem: y_pred.max(), {y_pred.max()}, y_true.max() {y_true.max()}")
                hd95 = 0

            return dsc, hd95

        def get_score(file_ref, file_pred, ref_reader, pred_reader):
            # print(file_ref+('+++++++++++'.center(20, "="))+file_pred)
            # print("++++".center(50, "="))
            seg_ref, seg_ref_dict = ref_reader.read_seg(seg_fname=file_ref)
            seg_pred, seg_pred_dict = pred_reader.read_seg(file_pred)

            seg_ref, seg_pred = seg_ref.squeeze(axis=0), seg_pred.squeeze(axis=0)

            # (C, H, W)
            assert seg_ref.shape == seg_pred.shape, f"invalid shape, seg: {seg_pred.shape}, ref: {seg_ref.shape}"

            # seg_ref = torch.tensor(seg_ref, dtype=torch.int64)  # .cuda(1)  # (C, H, W)
            # seg_pred = torch.tensor(seg_pred, dtype=torch.int64)  # .cuda(1)  # (C, H, W)

            case_vol_dice, case_hd95 = dice_score_vol(seg_pred, seg_ref, num_classes=4)

            return case_hd95, case_vol_dice

        # self.print_to_log_file(f"starting...")
        reader = self.plans_manager.image_reader_writer_class()
        if self.debug:
            files_ref, files_pred = files_ref[:1], files_pred[:1]

        results = Parallel(-1, prefer="threads")(delayed(get_score)(file_ref, file_pred, ref_reader, pred_reader)
                           for file_ref, file_pred, ref_reader, pred_reader in zip(
            files_ref[:1], files_pred[:1],
            [reader] * len(files_ref),
            [reader] * len(files_pred)
        ))
        # self.print_to_log_file(f"aggregating...")
        for res in results:
            # all_dice_slice_level.append(res[0])  # [(C1, 4), (C2, 4) ...]
            if res[0] != 0:
                all_haus95.append(res[0])
            all_dice_vol_level.append(res[1])  # [(4,), (4,) ...]
        self.print_to_log_file(f"starting computing scores...")

        all_dice_vol_level = np.array(all_dice_vol_level)  # (N)

        dsc_v = np.mean(all_dice_vol_level, axis=0)  # (3, )

        haus95 = np.array(all_haus95)  # (N)

        dsc_v_mean = dsc_v

        haus95_mean = np.mean(haus95, axis=0)

        # print(f"=============={dsc_v.shape, haus95_mean.shape}==================")

        if self.local_rank == 0:
            self.print_to_log_file(f"DSC_v: {dsc_v_mean:.2%}")

            if not self.debug:
                # wandb.log(data={"test/DSC": dsc_mean},step=self.current_epoch)
                wandb.log(data={"test/DSC": dsc_v_mean},step=self.current_epoch)
                wandb.log(data={"test/hd95": haus95_mean}, step=self.current_epoch)

        self.print_to_log_file(f"finished real validation")
        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()

    def initialize(self):
        self.initialize_wandb(name="Lung", project="Lung")
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            if self.network_config is None:
                self.network = self.build_network_architecture(self.plans_manager, self.dataset_json,
                                                               self.configuration_manager,
                                                               self.num_input_channels,
                                                               enable_deep_supervision=True)  # .to(self.device)
            else:
                self.network, self.network_config = self.build_network_architecture(**self.network_config)

            self.network = self.network.to(self.device)
            # compile network for free speedup
            if ('nnUNet_compile' in os.environ.keys()) and (
                    os.environ['nnUNet_compile'].lower() in ('true', '1', 't')):
                self.print_to_log_file('Compiling network...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                    "wandb_id": self.id,
                    "network_config": self.network_config,
                    "initial_lr": self.initial_lr,
                    # "backbone_lr": self.backbone_lr,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

