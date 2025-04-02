# Instant-NSR Training
WORKSPACE="./work_dirs"
mkdir -p $WORKSPACE
EPOCH=1000
EVAL_INTERVAL=100
ANNO_POINT=1
AGG_POINT_NUM=5
PRETRAIN=20
ITERATIVE=100
PSEUDO_INTER=20
SAMTHR=1.0
INSTANCE_LOSS_WEIGHT=0.1
MODE="train"
SCENE="dtu_scan55"
EXPNAME="${SCENE}_$(date +%Y%m%d_%H%M%S)"
WORKSPACE="$WORKSPACE/$EXPNAME"
mkdir -p $WORKSPACE
DATASETS="./datasets/dtu/${SCENE}"
SAM_FEAT_PATH="./datasets/dtu/${SCENE}/sam_feat"
ANNO_PATH="./datasets/dtu/${SCENE}/anno_box.json"

CUDA_VISIBLE_DEVICES=6 python train_nerf.py \
--path $DATASETS  \
--workspace $WORKSPACE \
--epoch $EPOCH \
--eval_interval $EVAL_INTERVAL \
--mode $MODE \
--sam_ckp_path "./sam_vit_h_4b8939.pth" \
--anno_path $ANNO_PATH \
--anno_point $ANNO_POINT \
--instance_loss $INSTANCE_LOSS_WEIGHT \
--pretrain_epoch $PRETRAIN \
--iterative $ITERATIVE \
--update_pseudo_label_interval $PSEUDO_INTER \
--agg_point_num $AGG_POINT_NUM \
--sam_thr $SAMTHR \
--sam_feat_path $SAM_FEAT_PATH 