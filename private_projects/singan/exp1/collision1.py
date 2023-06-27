default_scope = 'mmedit'
randomness = dict(seed=2022, diff_rank_seed=True)
dist_params = dict(backend='nccl')
opencv_num_threads = 0
mp_start_method = 'fork'
default_hooks = dict(
    timer=dict(type='EditIterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10000,
        by_epoch=False,
        max_keep_ckpts=20,
        less_keys=['FID-Full-50k/fid', 'FID-50k/fid', 'swd/avg'],
        greater_keys=['IS-50k/is', 'ms-ssim/avg'],
        save_optimizer=True))
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_level = 'INFO'
log_processor = dict(type='EditLogProcessor', by_epoch=False)
load_from = None
resume = None
model_wrapper_cfg = dict(
    type='MMSeparateDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=True)
vis_backends = [dict(type='GenVisBackend')]
visualizer = dict(
    type='GenVisualizer', vis_backends=[dict(type='GenVisBackend')])
train_cfg = dict(
    by_epoch=False, val_begin=1, val_interval=10000, max_iters=33000)
val_cfg = None
val_evaluator = None
test_cfg = None
test_evaluator = None
optim_wrapper = dict(
    constructor='SinGANOptimWrapperConstructor',
    generator=dict(optimizer=dict(type='Adam', lr=0.0005, betas=(0.5, 0.999))),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.0005, betas=(0.5, 0.999))))
num_scales = 10
generator_steps = 3
discriminator_steps = 3
iters_per_scale = 1000
test_pkl_data = None
model = dict(
    type='SinGAN',
    data_preprocessor=dict(
        type='EditDataPreprocessor', non_image_keys=['input_sample']),
    generator=dict(
        type='SinGANMultiScaleGenerator',
        in_channels=1,
        out_channels=1,
        num_scales=10),
    discriminator=dict(
        type='SinGANMultiScaleDiscriminator', in_channels=1, num_scales=10),
    noise_weight_init=0.1,
    test_pkl_data=None,
    lr_scheduler_args=dict(milestones=[1600], gamma=0.1),
    generator_steps=3,
    discriminator_steps=3,
    iters_per_scale=1000,
    num_scales=10)
min_size = 45
max_size = 800
dataset_type = 'SinGANDataset'
data_root = 'data/demo/000000000000200106_4_3_TA07_02_20210903115617869_00_640_640_1440_1440.jpg'
pipeline = [
    dict(
        type='PackEditInputs',
        keys=[
            'real_scale0', 'real_scale1', 'real_scale2', 'real_scale3',
            'real_scale4', 'real_scale5', 'real_scale6', 'real_scale7',
            'real_scale8', 'real_scale9', 'real_scale10', 'input_sample'
        ])
]
dataset = dict(
    type='SinGANDataset',
    data_root=
    'data/demo/000000000000200106_4_3_TA07_02_20210903115617869_00_640_640_1440_1440.jpg',
    min_size=45,
    max_size=800,
    scale_factor_init=0.9,
    pipeline=[
        dict(
            type='PackEditInputs',
            keys=[
                'real_scale0', 'real_scale1', 'real_scale2', 'real_scale3',
                'real_scale4', 'real_scale5', 'real_scale6', 'real_scale7',
                'real_scale8', 'real_scale9', 'real_scale10', 'input_sample'
            ])
    ])
train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=dict(
        type='SinGANDataset',
        data_root=
        'data/demo/000000000000200106_4_3_TA07_02_20210903115617869_00_640_640_1440_1440.jpg',
        min_size=45,
        max_size=800,
        scale_factor_init=0.9,
        pipeline=[
            dict(
                type='PackEditInputs',
                keys=[
                    'real_scale0', 'real_scale1', 'real_scale2', 'real_scale3',
                    'real_scale4', 'real_scale5', 'real_scale6', 'real_scale7',
                    'real_scale8', 'real_scale9', 'real_scale10',
                    'input_sample'
                ])
        ]),
    sampler=None,
    persistent_workers=False)
total_iters = 33000
custom_hooks = [
    dict(
        type='PickleDataHook',
        output_dir='pickle',
        interval=-1,
        after_run=True,
        data_name_list=['noise_weights', 'fixed_noises', 'curr_stage']),
    dict(
        type='GenVisualizationHook',
        interval=33000,
        fixed_input=True,
        n_samples=625,
        n_row=25,
        vis_kwargs_list=dict(type='SinGAN', name='fish'))
]
launcher = 'none'
work_dir = 'private_projects/singan/exp1'
