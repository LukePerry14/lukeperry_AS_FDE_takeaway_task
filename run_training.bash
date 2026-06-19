#!/bin/sh

for flag in "" "--tranches"; do
    python "$(dirname "$0")/main.py" $flag
done

read -p "Press enter to continue..."

