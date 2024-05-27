CUDA_VISIBLE_DEVICES=2 python moco_cifar10.py  --batch_size=512  &
sleep 3
CUDA_VISIBLE_DEVICES=3 python moco_cifar10.py  --batch_size=1024