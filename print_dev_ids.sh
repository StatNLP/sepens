#!/bin/bash


if [ -z "$1" ]
then
  echo "USAGE . ./print_dev_ids.sh splitnum"
else

split=$1

list_spc_ns=`grep '	0	' dev.dat | cut -f1 | uniq | tr "\n" ' '`
list_spc_ss=`grep '	1	' dev.dat | cut -f1 | uniq | tr "\n" ' '`

list_comma_ns=`grep '	0	' dev.dat | cut -f1 | uniq | tr "\n" ','`
list_comma_ss=`grep '	1	' dev.dat | cut -f1 | uniq | tr "\n" ','`

list_spc=$"$list_spc_ns\\\n$list_spc_ss"

echo -e "$list_spc_ns\n$list_spc_ss"
echo -e "$list_spc_ns\n$list_spc_ss" > ./dev_ids_s${split}.dat
echo -e "$list_spc_ns\n$list_spc_ss" > ./dev_ids.dat
sed "s/ /\n/g" dev_ids.dat | sed -r '/^\s*$/d' > ./dev_ids.lst
echo "Written to dev_ids_s${split}.dat"

echo -e "$list_comma_ns\n$list_comma_ss"
echo -e "$list_comma_ns\n$list_comma_ss" > ./dev_ids_s${split}.inc
echo -e "$list_comma_ns\n$list_comma_ss" > ./dev_ids.inc
echo -e "valid_ns = {$list_comma_ns}\nvalid_ss = {$list_comma_ss}" > ./dev_ids.py
echo "Written to ./dev_ids_s${split}.inc and dev_ids.py."

fi


