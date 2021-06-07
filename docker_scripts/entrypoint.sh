#!bin/bash

set -e

echo "password: $PASS"

# start ssh server
sudo service ssh start
# starting tigervncserver using password: $password
tigervncserver
export DISPLAY=:1

exec "$@"
