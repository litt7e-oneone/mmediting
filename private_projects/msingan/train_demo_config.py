_base_ = ['../../configs/_base_/gen_default_runtime.py']

# MODEL WRAPPER
model_wrapper_cfg = dict(find_unused_parameters=True)

# MODEL
input_channels = 1  # 3 for 'color' ; 1 for 'grayscale'
num_scales = 10  # start from zero
generator_steps = 3
discriminator_steps = 3
iters_per_scale = 2000
# NOTE: add by user, e.g.:
# test_pkl_data = ('./data/singan/singan_fis_20210406_201006-860d91b6.pkl')
test_pkl_data = None

model = dict(
    type='MSinGAN',
    data_preprocessor=dict(
        type='EditDataPreprocessor', non_image_keys=['input_sample', 'input_index']),
    generator=dict(
        type='MSinGANMultiScaleGenerator',
        in_channels=input_channels,
        out_channels=input_channels,
        num_scales=num_scales,
    ),
    discriminator=dict(
        type='MSinGANMultiScaleDiscriminator',
        in_channels=input_channels,
        num_scales=num_scales,
    ),
    noise_weight_init=0.1,
    test_pkl_data=test_pkl_data,
    lr_scheduler_args=dict(milestones=[1600], gamma=0.1),
    generator_steps=generator_steps,
    discriminator_steps=discriminator_steps,
    iters_per_scale=iters_per_scale,
    num_scales=num_scales)

# DATA
min_size = 64
max_size = 800
dataset_type = 'MSinGANDataset'
data_root = './data/demo'

pipeline = [
    dict(
        type='PackEditInputsWithIndex',
        keys=[f'real_scale{i}' for i in range(num_scales + 1)] + ['input_sample'])
]
dataset = dict(
    type=dataset_type,
    data_root=data_root,
    min_size=min_size,
    max_size=max_size,
    scale_factor_init=0.75,
    pipeline=pipeline,
    image_format='grayscale')  # either 'color' or 'grayscale'

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=dataset,
    sampler= dict(type='InfiniteSampler', shuffle=True),  # dict(type='InfiniteSampler', shuffle=True)
    persistent_workers=False)

# TRAINING
optim_wrapper = dict(
    constructor='SinGANOptimWrapperConstructor',
    generator=dict(optimizer=dict(type='Adam', lr=0.0005, betas=(0.5, 0.999))),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.0005, betas=(0.5, 0.999))))

total_iters = (num_scales + 1) * iters_per_scale * discriminator_steps
train_cfg = dict(max_iters=total_iters)

# HOOK
custom_hooks = [
    dict(
        type='PickleDataHook',
        output_dir='pickle',
        interval=-1,
        after_run=True,
        data_name_list=['noise_weights', 'fixed_noises', 'curr_stage']),
    dict(
        type='GenVisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=dict(type='SinGAN', name='DEMO'))
]

# NOTE: SinGAN do not support val_loop and test_loop, please use
# 'tools/utils/inference_singan.py' to evaluate and generate images.
val_cfg = test_cfg = None
val_evaluator = test_evaluator = None
