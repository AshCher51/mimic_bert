nohup python preprocessing.py
mv nohup.out preprocessing.out
nohup python actual_train.py --dev False --epochs 20
mv nohup.out training_output.out
