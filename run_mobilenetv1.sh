source activate tc
a=$(echo "1e-4"| awk '{printf("%f",$0)}')
alg=$'SGD_mcp'
for i in {1..5}
do
    bsub -J $alg$(echo| awk "{print $i*$a}") -e $alg$(echo| awk "{print $i*$a}").err -o $alg$(echo| awk "{print $i*$a}").out -n 1 -q volta -gpu "num=1:mode=exclusive_process"\
    python run.py --model mobilenetv1 --optimizer $alg --lambda_ $(echo| awk "{print $i*$a}") --max_epoch 50 --batch_size 128 --learning_rate 0.01 --edition 2.0 --Np 1 --Np2 1
sleep 10
done


