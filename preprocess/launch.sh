#!/bin/bash

# run script in a daemon screen to prevent stopping when the ssh connection is lost
# command: screen -dmS preprocessing bash launch.sh
# reminder: screen -XS <screen-id> quit
# reminder: "tail -f monitoring.txt" to track progression

echo -n "" > monitoring.txt
python3 -u preprocess.py >> monitoring.txt 2>&1
