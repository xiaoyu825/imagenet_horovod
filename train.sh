horovodrun -np 4 -H localhost:4 python train_kfold.py --distribute 'True'
#horovodrun -np 8 -H 10.207.43.19:4,10.207.43.20:4 python train_kfold.py --distribute 'True'
#horovodrun -np 8 -H 10.207.178.34:4,10.207.179.210:4 python3 train_kfold.py --distribute 'True'