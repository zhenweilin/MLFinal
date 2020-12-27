source activate tc
a=$(echo "1e-4"| awk '{printf("%f",$0)}')
for i in {1..10}
do
    bsub -J MLFfinaliter$(echo| awk "{print $i*$a}") -e SGD%J.err -o SGD%J.out -n 1 -q volta -gpu "num=1:mode=exclusive_process"\
    python run.py --model mobilenetv1 --optimizer SGD_l1 --lambda_ $(echo| awk "{print $i*$a}") --max_epoch 200 --batch_size 128 --learning_rate 0.1 --edition 2.0 --Np 1 --Np2 1
sleep 10
done


