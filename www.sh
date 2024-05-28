## basic
python moco_cifar10.py  --results_dir=basic --batch_size=128 --lr=0.02
python moco_cifar10.py --results_dir=basic  --batch_size=256 --lr=0.03



# GCL
# python moco_cifar10_GCL.py  --results_dir=gcl_2_128 --batch_size=128 --lr=0.03 --num_patch=2 --epochs=200 --cos=True --schedule=[] --symmetric=False
# python moco_cifar10_GCL.py  --results_dir=gcl_4_128 --batch_size=128 --lr=0.03 --num_patch=4 --epochs=200 --cos=True --schedule=[] --symmetric=False
# python moco_cifar10_GCL.py  --results_dir=gcl_8_128 --batch_size=128 --lr=0.03 --num_patch=8 --epochs=200 --cos=True --schedule=[] --symmetric=False

# python moco_cifar10_GCL.py  --results_dir=gcl_2_256 --batch_size=256 --lr=0.03 --num_patch=2 --epochs=200 --cos=True --schedule=[] --symmetric=False
# python moco_cifar10_GCL.py  --results_dir=gcl_4_256 --batch_size=256 --lr=0.03 --num_patch=4 --epochs=200 --cos=True --schedule=[] --symmetric=False
# python moco_cifar10_GCL.py  --results_dir=gcl_8_256 --batch_size=256 --lr=0.03 --num_patch=8 --epochs=200 --cos=True --schedule=[] --symmetric=False
