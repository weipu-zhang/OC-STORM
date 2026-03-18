#!/bin/bash
kill -9 $(ps -ef | grep "train.py" | awk '{print $2}')
kill -9 $(ps -ef | grep "train_async.py" | awk '{print $2}')
kill -9 $(ps -ef | grep "multiprocessing.spawn" | awk '{print $2}')
