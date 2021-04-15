# model settings
alphabet = list('0123456789/-:.年月日 ')
nclass = len(alphabet) + 1
text_max_len = 25

model = dict(
    type='SAR',
    pretrained=None,
    nh=512,
    nclass=nclass,
    alphabet=alphabet,
    text_max_len=text_max_len,
    show_attention=False,
    training=True
    )

# model training and testing settings
train_cfg = dict(
    neck=None)
test_cfg = dict(
    neck=None)

# dataset settings
dataset_type = 'RecognizerDataset'
data_root = '/'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations_SAR'),
    dict(type='RandomBlur', p=0.2),
    dict(type='BrightnessContrastSaturation', p=0.3),
    dict(type='RandomImgCompression', quality=[60, 80], p=0.3),
    dict(type='Resize_SAR', img_scale=(48,256)),
    dict(type='DefaultFormatBundle_SAR', alphabet=alphabet, text_max_len=text_max_len),
    dict(type='Collect_SAR', keys=['img', 'target_variable', 'target_cp', 'mask']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(48,256),
        flip=False,
        transforms=[
            dict(type='Resize_SAR', img_scale=(48,256)),
            dict(type='DefaultFormatBundle_SAR', alphabet=alphabet, text_max_len=text_max_len, training=False),
            dict(type='Collect_SAR', keys=['img', 'target_variable', 'mask']),
        ])
]
data = dict(
    imgs_per_gpu=128,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'labels.txt',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'labels.txt',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'labels.txt',
        img_prefix=data_root,
        pipeline=test_pipeline))
# evaluation = dict(interval=1, metric='bbox')
# optimizer
optimizer = dict(type='Adadelta', lr=1)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=1.0/3,
    step=[10, 15])
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 200
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/sar50'
load_from = None
resume_from = None
workflow = [('train', 1)]
