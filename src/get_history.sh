cp ../data/ICEWS14/ent2word.py ../data/ICEWS18/
cp ../data/ICEWS14/ent2word.py ../data/ICEWS05-15/
cd ../data/ICEWS14/ || exit
python ent2word.py
cd ../ICEWS18/ || exit
python ent2word.py
cd ../ICEWS05-15/ || exit
python ent2word.py
cd ../../src || exit
#nohup python get_history.py --dataset ICEWS14 --method once > ./logfile/ICEWS14_once.log &
#nohup python get_history.py --dataset ICEWS18 --method once > ./logfile/ICEWS18_once.log &
#nohup python get_history.py --dataset ICEWS05-15 --method once > ./logfile/ICEWS05-15_once.log &
#nohup python get_history.py --dataset WIKI --method once > ./logfile/WIKI_once.log &
#nohup python get_history.py --dataset YAGO --method once > ./logfile/YAGO_once.log &
#nohup python get_history.py --dataset GDELT --method once > ./logfile/GDELT_once.log &
#
#nohup python get_history.py --dataset ICEWS14 --method time_decay > ./logfile/ICEWS14_time_decay.log &
#nohup python get_history.py --dataset ICEWS18 --method time_decay > ./logfile/ICEWS18_time_decay.log &
#nohup python get_history.py --dataset ICEWS05-15 --method time_decay > ./logfile/ICEWS0515_time_decay.log &
#nohup python get_history.py --dataset WIKI --method time_decay > ./logfile/WIKI_time_decay.log &
#nohup python get_history.py --dataset YAGO --method time_decay > ./logfile/YAGO_time_decay.log &
#nohup python get_history.py --dataset GDELT --method time_decay > ./logfile/GDELT_time_decay.log &

#nohup python get_history.py --dataset ICEWS14 --method time_hawk_edge --num-workers 4 > ./logfile/ICEWS14_time_decay.log &
#nohup python get_history.py --dataset ICEWS18 --method time_hawk_edge --num-workers 4 > ./logfile/ICEWS18_time_decay.log &
#nohup python get_history.py --dataset ICEWS05-15 --method time_hawk_edge --num-workers 8 > ./logfile/ICEWS0515_time_decay.log &
#nohup python get_history.py --dataset WIKI --method time_hawk_edge --num-workers 4 > ./logfile/WIKI_time_decay.log &
#nohup python get_history.py --dataset YAGO --method time_hawk_edge --num-workers 8 > ./logfile/YAGO_time_decay.log &
#nohup python get_history.py --dataset GDELT --method time_hawk_edge --num-workers 8 > ./logfile/GDELT_time_decay.log &


nohup python get_history.py --dataset ICEWS14 --method time_hawk_edge_2 --num-workers 8 > ./logfile/ICEWS14_history.log &

nohup python get_history.py --dataset ICEWS18 --method time_hawk_edge_2 --num-workers 2 > ./logfile/ICEWS18_history.log &

nohup python get_history.py --dataset ICEWS05-15 --method time_hawk_edge_2 --num-workers 16 > ./logfile/ICEWS0515_history.log &

nohup python get_history.py --dataset WIKI --method time_hawk_edge_2 --num-workers 8 > ./logfile/WIKI_history.log &

nohup python get_history.py --dataset YAGO --method time_hawk_edge_2 --num-workers 8 > ./logfile/YAGO_history.log &

nohup python get_history.py --dataset GDELT --method time_hawk_edge_2 --num-workers 16 > ./logfile/GDELT_history.log &