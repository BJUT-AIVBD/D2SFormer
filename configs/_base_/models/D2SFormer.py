# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='CSWin',
        embed_dim=64,
        patch_size=4,
        depth=[1, 2, 21, 1],
        num_heads=[2,4,8,16],
        split_size=[1,2,7,7],
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1),
    decode_head=dict(
        type='BifpnHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
