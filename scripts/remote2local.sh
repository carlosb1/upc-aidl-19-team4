#!/bin/bash

PASS="olakase"
CONFIG_FILE="/content/disk/practica_ai"

if ! [ -x "$(command -v sshpass)" ] && ! [ -x "$(command -v rsync)" ]; then
	echo "error: install sshpass and rsync"
	exit 1
fi


if [[ $# -ne 2 ]]; then
	echo "error: local2remote.sh USER:HOST PORT"
	exit 1
fi

EXCLUDE_FILES="--exclude .git --exclude data_tmp --exclude datasets"
sync="sshpass -p \"$PASS\" rsync $EXCLUDE_FILES -av -e 'ssh -p ${2}' $1:\"$CONFIG_FILE\" ./ --ignore-errors"  #--delete --ignore-errors 2>./rsync.err" 
eval $sync
