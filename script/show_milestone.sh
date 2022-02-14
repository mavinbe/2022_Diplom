#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MILESTONE_TEXT=$(cat "$SCRIPT_DIR/current_milestone.txt")
zenity --info --ellipsize --text="#MILESTONE \n$MILESTONE_TEXT"