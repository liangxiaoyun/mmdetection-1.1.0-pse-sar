# model settings
n = 6
text_classify=['text']
result_num=len(text_classify)*n
model = dict(
    type='PSENet',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        style='pytorch'),
    neck=dict(type='PSEFPN',
              result_num=result_num,
              scale=1)
    )

# model training and testing settings
train_cfg = dict(
    neck=dict(
        result_num=result_num,
        scale=1)
    )
test_cfg = dict(
    neck=dict(
        result_num=result_num,
        scale=1))

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/Users/duoduo/Desktop/天池图片篡改检测/s2_data/data/small_train/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations_PSE', result_num=len(text_classify), n=n, m=0.5, with_poly_mask=True),
    # dict(type='Randomrotate', angle_range=20, rotate_ratio=0.5),
    # dict(type='Randomrotate', angle_range=[90, 180, 270], rotate_ratio=0.5, mode='fix'),
    dict(type='Resize', img_scale=[(1280,1280)], multiscale_mode='value', keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640,640)),
    dict(type='RandomFlip', flip_ratio=0.2),
    dict(type='RandomBlur', p=0.3),
    dict(type='BrightnessContrastSaturation', p=0.3),
    dict(type='RandomImgCompression', quality=[60, 80], p=0.3),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'score_maps', 'training_mask']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 1280),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
# optimizer
optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=1.0/3,
    step=[400, 700, 900])
checkpoint_config = dict(interval=50)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 1000
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/psenet_r50'
load_from = None
resume_from = None
workflow = [('train', 1)]
