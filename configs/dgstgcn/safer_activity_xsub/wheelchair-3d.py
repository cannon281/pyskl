modality = 'j'
graph = 'coco'
work_dir = f'./work_dirs/dgstgcn/safer_activity_xsub/wheelchair-3d/{modality}'

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='DGSTGCN',
        gcn_ratio=0.125,
        gcn_ctr='T',
        gcn_ada='T',
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
        graph_cfg=dict(layout=graph, mode='random', num_filter=8, init_off=.04, init_std=.02)),
    cls_head=dict(type='GCNHead', num_classes=15, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = 'Pkl/aic_wheelchair_dataset_with_3d.pkl'

# Skip PreNormalize3D since 3D data is already normalized (following STGCN++ pattern)
train_pipeline = [
    dict(type='GenSkeFeat', dataset='coco', feats=[modality]),
    dict(type='SampleSequentialFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='GenSkeFeat', dataset='coco', feats=[modality]),
    dict(type='SampleSequentialFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='GenSkeFeat', dataset='coco', feats=[modality]),
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
            use_3d_keypoints=True)),  # Enable 3D keypoints
    val=dict(
        type=dataset_type, 
        ann_file=ann_file, 
        pipeline=val_pipeline, 
        split='sub_test', 
        preprocess_type="sequential",
        use_3d_keypoints=True),
    test=dict(
        type=dataset_type, 
        ann_file=ann_file, 
        pipeline=test_pipeline, 
        split='sub_test', 
        preprocess_type="sequential",
        use_3d_keypoints=True))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])