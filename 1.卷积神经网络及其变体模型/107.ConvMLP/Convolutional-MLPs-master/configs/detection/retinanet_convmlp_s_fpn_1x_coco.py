_base_ = [
    './_base_/models/retinanet_r50_fpn.py',
    './_base_/datasets/coco_detection.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py'
]

model = dict(
    pretrained='https://shi-labs.com/projects/convmlp/checkpoints/convmlp_s_imagenet.pth',
    backbone=dict(type='DetConvMLPSmall'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=5))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
