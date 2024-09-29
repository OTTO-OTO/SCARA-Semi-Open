#!bin/bash
LOG_NAME=./Retrain-law-0904.log
# Run the application
rm $LOG_NAME
# export NCCL_P2P_DISABLE=1

screen -L -Logfile $LOG_NAME python LegalTest.py
# screen -L -Logfile $LOG_NAME python LegalTest.py