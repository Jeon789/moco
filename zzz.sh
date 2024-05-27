python main_moco.py \
  -a resnet50 \
  --gpu 0 \
  --lr 0.03 \
  --batch-size 512 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  /data/jan4021/Data/ImageNet
#   [your imagenet-folder with train and val folders]