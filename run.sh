#/bin/bash
#BSUB -J MLFinal
#BSUB -e %J.err
#BSUB -o %J.out
#BSUB -n 1 
#BSUB -q volta
#BSUB -gpu "num=1:mode=exclusive_process"
source activate tc

python run.py --model mobilenetv1 --optimizer SGD_l1 --lambda_ 0.0001 --max_epoch 200 --batch_size 128 --learning_rate 0.1 --edition 2.0 --Np 1 --Np2 1
python run.py --model mobilenetv1 --optimizer SGD_l1 --lambda_ 0.0002 --max_epoch 200 --batch_size 128 --learning_rate 0.1 --edition 2.1 --Np 1 --Np2 1
python run.py --model mobilenetv1 --optimizer SGD_l1 --lambda_ 0.0003 --max_epoch 200 --batch_size 128 --learning_rate 0.1 --edition 2.2 --Np 1 --Np2 1
python run.py --model mobilenetv1 --optimizer SGD_l1 --lambda_ 0.0004 --max_epoch 200 --batch_size 128 --learning_rate 0.1 --edition 2.3 --Np 1 --Np2 1
python run.py --model mobilenetv1 --optimizer SGD_l1 --lambda_ 0.0005 --max_epoch 200 --batch_size 128 --learning_rate 0.1 --edition 2.4 --Np 1 --Np2 1
python run.py --model mobilenetv1 --optimizer SGD_l1 --lambda_ 0.0006 --max_epoch 200 --batch_size 128 --learning_rate 0.1 --edition 2.5 --Np 1 --Np2 1
python run.py --model mobilenetv1 --optimizer SGD_l1 --lambda_ 0.0007 --max_epoch 200 --batch_size 128 --learning_rate 0.1 --edition 2.6 --Np 1 --Np2 1
python run.py --model mobilenetv1 --optimizer SGD_l1 --lambda_ 0.0008 --max_epoch 200 --batch_size 128 --learning_rate 0.1 --edition 2.7 --Np 1 --Np2 1
python run.py --model mobilenetv1 --optimizer SGD_l1 --lambda_ 0.0009 --max_epoch 200 --batch_size 128 --learning_rate 0.1 --edition 2.8 --Np 1 --Np2 1
python run.py --model mobilenetv1 --optimizer SGD_l1 --lambda_ 0.00025 --max_epoch 200 --batch_size 128 --learning_rate 0.1 --edition 2.9 --Np 1 --Np2 1

