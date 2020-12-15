# BDRL
The source code of the paper "BDRL: Combining Bert and Deep Reinforcement Learning to learn the Combinatorial Optimization Algorithms over Graphs"
# Data
Most of the data sets in the experiment are generated data, such as VRP, TSP, etc.
# How to run
BDRL can be divided into three independent models: BDRL as a whole, BDRL without fine-tuning, and BERT fine-tuning only. The X in the parameter can be set as needed.
# Generating data
python generate_data.py --Data all --name validation --seed python generate_data.py --Data all --name test --seed
# Tranining
python train.py --size=X --epoch=X --batch_size=X --train_size=X --val_size=X --lr=X
# Test
python test_random.py --size=X --batch_size=X --test_size=X --test_steps=X
# Fine-tuning
# VRP
python run_bert_classifier.py --task_name co --do_train --do_eval --do_predict --data_dir ./data/vrp --bert_model bert-base-uncased --max_seq_length 20 --train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 3.0 --output_dir ./output_vrp/ --gradient_accumulation_steps 1 --eval_batch_size 512
# TSP
python run_bert_classifier.py --task_name co --do_train --do_eval --do_predict --data_dir ./data/tsp --bert_model bert-base-cased --max_seq_length 200 --train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 3.0 --output_dir ./output_tsp/ --gradient_accumulation_steps 1 --eval_batch_size 512
