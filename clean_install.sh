#!/usr/bin/env bash
# set -e -x

read -p "Are you sure you want to delete all compiled files for a clean installation? (Y/n): " yesno

if [ "${yesno,,}" == "y" ]; then
  echo "Removing nighresjava"; rm -rf nighresjava/
  echo "Removing cbstools-public"; rm -rf cbstools-public/
  echo "Removing imcn-imaging"; rm -rf imcn-imaging/
  echo "Removing nighres.egg-info"; rm -rf nighres.egg-info/
  echo "Removing nighres_examples"; rm -rf nighres_examples/
elif [ "${yesno,,}" == "n" ]; then
  echo "Not cleaning"
else
  echo "Invalid input"
fi
