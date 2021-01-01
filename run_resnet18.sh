source activate tc
a=$(echo "1e-4"| awk '{printf("%f",$0)}')
alg=$'SGD_l1'
model=$'resnet18'
for i in {2..10}
do
    bsub -J $slg$(echo| awk "{print $i*$a}") -e $slg$(echo| awk "{print $i*$a}").err -o $slg$(echo| awk "{print $i*$a}").out -n 1 -q volta -gpu "num=1:mode=exclusive_process"\
    python run.py --model $model --optimizer $alg --lambda_ $(echo| awk "{print $i*$a}") --max_epoch 200 --batch_size 128 --learning_rate 0.1 --edition 2.0 --Np 1 --Np2 1
sleep 10
done