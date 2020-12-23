#/bin/bash
#BSUB -J MLFinal
#BSUB -e SGD%J.err
#BSUB -o SGD%J.out
#BSUB -n 3 
#BSUB -q volta
#BSUB -gpu "num=3:mode=exclusive_process"
source activate tc

python run.py --model mobilenetv1 --optimizer SGD_l1 --lambda_ 0.00015 --max_epoch 200 --batch_size 128 --learning_rate 0.1 --edition 2.0 --Np 1 --Np2 1 &
python run.py --model mobilenetv1 --optimizer SGD_l1 --lambda_ 0.00025 --max_epoch 200 --batch_size 128 --learning_rate 0.1 --edition 2.1 --Np 1 --Np2 1 & 
python run.py --model mobilenetv1 --optimizer SGD_l1 --lambda_ 0.00035 --max_epoch 200 --batch_size 128 --learning_rate 0.1 --edition 2.2 --Np 1 --Np2 1 &
python run.py --model mobilenetv1 --optimizer SGD_l1 --lambda_ 0.00045 --max_epoch 200 --batch_size 128 --learning_rate 0.1 --edition 2.3 --Np 1 --Np2 1 &
python run.py --model mobilenetv1 --optimizer SGD_l1 --lambda_ 0.00055 --max_epoch 200 --batch_size 128 --learning_rate 0.1 --edition 2.4 --Np 1 --Np2 1 &
python run.py --model mobilenetv1 --optimizer SGD_l1 --lambda_ 0.00065 --max_epoch 200 --batch_size 128 --learning_rate 0.1 --edition 2.5 --Np 1 --Np2 1 &
python run.py --model mobilenetv1 --optimizer SGD_l1 --lambda_ 0.00075 --max_epoch 200 --batch_size 128 --learning_rate 0.1 --edition 2.6 --Np 1 --Np2 1 &
python run.py --model mobilenetv1 --optimizer SGD_l1 --lambda_ 0.00085 --max_epoch 200 --batch_size 128 --learning_rate 0.1 --edition 2.7 --Np 1 --Np2 1 &
python run.py --model mobilenetv1 --optimizer SGD_l1 --lambda_ 0.00095 --max_epoch 200 --batch_size 128 --learning_rate 0.1 --edition 2.8 --Np 1 --Np2 1 &