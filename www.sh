## basic
# python moco_cifar10.py  --results_dir=basic_128  --batch_size=128 --lr=0.02
# python moco_cifar10.py  --results_dir=basic_256  --batch_size=256 --lr=0.03




# GCL
# python moco_cifar10_GCL.py  --results_dir=gcl_2_128 --batch_size=128 --lr=0.03 --num_patch=2 --epochs=200 --cos=True --schedule=[] --symmetric=False
# python moco_cifar10_GCL.py  --results_dir=gcl_4_128 --batch_size=128 --lr=0.03 --num_patch=4 --epochs=200 --cos=True --schedule=[] --symmetric=False
# python moco_cifar10_GCL.py  --results_dir=gcl_8_128 --batch_size=128 --lr=0.03 --num_patch=8 --epochs=200 --cos=True --schedule=[] --symmetric=False

# python moco_cifar10_GCL.py  --results_dir=gcl_2_256 --batch_size=256 --lr=0.03 --num_patch=2 --epochs=200 --cos=True --schedule=[] --symmetric=False
# python moco_cifar10_GCL.py  --results_dir=gcl_4_256 --batch_size=256 --lr=0.03 --num_patch=4 --epochs=200 --cos=True --schedule=[] --symmetric=False
# python moco_cifar10_GCL.py  --results_dir=gcl_8_256 --batch_size=256 --lr=0.03 --num_patch=8 --epochs=200 --cos=True --schedule=[] --symmetric=False


# python moco_cifar10_GCL.py  --results_dir=dum --batch_size=768 --lr=0.03 --num_patch=4 --epochs=200 --cos=True --schedule=[] --symmetric=False --multi_gpu=[0,2,3]
# python moco_cifar10_GCL.py  --results_dir=dum --batch_size=256 --lr=0.03 --num_patch=4 --epochs=200 --cos=True --schedule=[] --symmetric=False --fm_method=3


CUDA_VISIBLE_DEVICES=0 python moco_cifar10_GCL.py  --results_dir=gcl_cos_4_256_fm1 --batch_size=256 --lr=0.03 --num_patch=4 --epochs=800 --cos=True --schedule=[] --symmetric=False --fm_method=1
CUDA_VISIBLE_DEVICES=2 python moco_cifar10_GCL.py  --results_dir=gcl_cos_4_256_fm2 --batch_size=256 --lr=0.03 --num_patch=4 --epochs=800 --cos=True --schedule=[] --symmetric=False --fm_method=2
CUDA_VISIBLE_DEVICES=3 python moco_cifar10_GCL.py  --results_dir=gcl_cos_4_256_fm3 --batch_size=256 --lr=0.03 --num_patch=4 --epochs=800 --cos=True --schedule=[] --symmetric=False --fm_method=3
