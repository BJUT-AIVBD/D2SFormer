_base_ = [
    '../_base_/models/D2SFormer.py', '../_base_/datasets/vaihingen.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    backbone=dict(
        type='DAT',
        pretrained='/home/bdaksh/Documents/yy/mmsegmentation-0.x/pretained/cswin/cswin_small_224.pth',
        embed_dim=64,
        depth=[2,4,32,2],
        num_heads=[2,4,8,16],
        split_size=[1,2,7,7],
        drop_path_rate=0.4,
        compress_ratio=3,
        gamma=2,
        b=1
    ),
    decode_head=dict(
        type='DBFAHead',
        in_channels=[64,128,256,512],
        channels=128,
        num_classes=6
    )
)

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data=dict(samples_per_gpu=2)
