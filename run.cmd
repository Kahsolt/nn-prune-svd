@ECHO OFF

REM clean
python run.py -M resnet18
REM attack (few samples)
python run.py -M resnet18 -X -L 4


REM prune
python run.py -M resnet18 --r_w 0.1
python run.py -M resnet18 --r_w 0.2
python run.py -M resnet18 --r_w 0.3
python run.py -M resnet18 --r_w 0.3 --r_b 0.00001
python run.py -M resnet18 --r_w 0.3 --r_b 0.00001 --n_prec 5

REM prune + attack
python run.py -M resnet18 --r_w 0.3 --r_b 0.00001 --n_prec 5 -X
