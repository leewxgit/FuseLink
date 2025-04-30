#!/bin/bash
  
echo "before"
sudo lspci -vvv | grep -i acsctl

readarray -t plx_list <<< $(lspci | grep -i plx | cut -c 1-7)
for line in "${plx_list[@]}"
do
        sudo setpci -s $line f2a.w=0000
done

echo "after"
sudo lspci -vvv | grep -i acsctl
echo "done"
