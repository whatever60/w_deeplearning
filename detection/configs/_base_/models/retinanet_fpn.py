# ResNet backbone (ResNet50 here)
backbone = dict(
    type="resnet",
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    # Freeze the parameters of stem and the first stage. These parameters learns universal low level features.
    frozen_stages=1,
    norm_cfg=dict(type="bn", requires_grad=True),
    # All BatchNorms in the backbone stop to update of mean and variance.
    norm_eval=True,
    # In pytorch mode, downsampling takes place at the middle 3x3 conv of BottleNeck (at the first 1x1 conv in caffe mode).
    style="pytorch",
    init_cfg=dict(type="pretrained", checkpoint="torchvision://resnet50"),
)

# FPN neck
neck = dict(
    type="fpn",
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    # RetinaNet only uses feature maps from the last 3 stages of resnet backbone.
    start_level=1,
    # We output 5 groups of feature maps.
    num_outs=5,
    # 2 extra groups of feature maps come from the output of resnet backbone.
    add_extra_convs="on_input",
)

# bounding box head
anchor_generator = dict(
    type="anchorgenerator",
    octave_base_scale=4,
    scales_per_octave=3,
    ratios=[0.5, 1.0, 2.0],
    strides=[8, 16, 32, 64, 128],
)

bbox_coder = dict(
    type="deltaxywhbboxcoder", target_means=[0, 0, 0, 0], target_stds=[1, 1, 1, 1]
)

loss_cls = dict(
    type="focalloss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
)

bbox_head = dict(
    type="retinahead",
    # dataset specific
    num_classes=80,
    # number of output channels of FPN
    in_channels=256,
    # Each head has 4 conv layers.
    stacked_convs=4,
    # number of channels in the middle of heads
    feat_channels=256,
    anchor_generator=anchor_generator,
    bbox_coder=bbox_coder,
    loss_cls=loss_cls,
    loss_bbox=dict(type="l1loss", loss_weight=1.0),
)

# train and test setting
assigner = dict(
    type="maxiouassigner",
    pos_iou_thr=0.5,
    neg_iou_thr=0.4,
    min_pos_iou=0,
    ignore_iof_thr=-1,
)

train_cfg = dict(
    assigner=assigner,
    allowed_border=-1,
    pos_weight=-1,
    debug=False,
)

test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type="nms", iou_threshold=0.5),
    max_per_img=100,
)
