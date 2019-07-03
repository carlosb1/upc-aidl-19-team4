#!/bin/bash
PASS="olakase"
CONFIG_FILE="/root/project"

if ! [ -x "$(command -v sshpass)" ] && ! [ -x "$(command -v rsync)" ]; then
	echo "error: install sshpass and rsync"
	exit 1
fi


if [[ $# -ne 2 ]]; then
	echo "error: local2remote.sh USER:HOST PORT"
	exit 1
fi

EXCLUDE_FILES="--exclude .git --exclude data_tmp --exclude datasets --exclude backups"
sync="sshpass -p \"$PASS\" rsync $EXCLUDE_FILES -valrhu -e 'ssh -p ${2}' ./ $1:\"$CONFIG_FILE\""  #--delete --ignore-errors 2>./rsync.err" 
eval $sync
