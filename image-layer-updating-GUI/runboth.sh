#!/bin/sh

## Kill both with control-c
## From: https://superuser.com/questions/1118204/how-to-use-ctrlc-to-kill-all-background-processes-started-in-a-bash-script/1118226
trap 'kill $BGPID; exit' SIGINT
python3 server.py &
BGPID=$!
python3 -m http.server
