model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='MSG3D',
        graph_cfg=dict(layout='coco', mode='binary_adj')),  # Keep coco layout for H36M->COCO mapping
    cls_head=dict(type='GCNHead', num_classes=15, in_channels=384))

dataset_type = 'PoseDataset'
ann_file = 'Pkl/aic_wheelchair_dataset_with_3d.pkl'

# Skip PreNormalize3D since 3D data is already normalized
train_pipeline = [
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='SampleSequentialFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

val_pipeline = [
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='SampleSequentialFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

test_pipeline = [
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='SampleSequentialFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=8,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type, 
            ann_file=ann_file, 
            pipeline=train_pipeline, 
            split='sub_train', 
            preprocess_type="sequential",
            use_3d_keypoints=True)),  # Added flag to use 3D keypoints
    val=dict(
        type=dataset_type, 
        ann_file=ann_file, 
        pipeline=val_pipeline, 
        split='sub_test', 
        preprocess_type="sequential",
        use_3d_keypoints=True),  # Added flag to use 3D keypoints
    test=dict(
        type=dataset_type, 
        ann_file=ann_file, 
        pipeline=test_pipeline, 
        split='sub_test', 
        preprocess_type="sequential",
        use_3d_keypoints=True))  # Added flag to use 3D keypoints

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/msg3d/safer_activity_xsub/wheelchair-3d'