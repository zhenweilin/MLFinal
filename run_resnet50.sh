source activate tc
a=$(echo "1e-5"| awk '{printf("%f",$0)}')
alg=$'SGD_mcp'
for i in {5..9}
do
    bsub -J resnet50$(echo| awk "{print $i*$a}") -e resnet50$(echo| awk "{print $i*$a}").err -o resnet50$(echo| awk "{print $i*$a}").out -n 1 -q volta -gpu "num=1:mode=exclusive_process"\
    python run.py --model resnet50 --optimizer $alg --lambda_ $(echo| awk "{print $i*$a}") --max_epoch 50 --batch_size 128 --learning_rate 0.01 --edition 2.0 --Np 1 --Np2 1
sleep 10
done