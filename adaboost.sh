#BSUB -J adaboost
#BSUB -e %J.err
#BSUB -o %J.out
#BSUB -n 1
activate tc
sleep 3600*2
python multi_adaboost.py