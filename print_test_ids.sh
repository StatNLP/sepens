#!/bin/bash


if [ -z "$1" ]
then
  echo "USAGE . ./print_test_ids.sh splitnum"
else

split=$1

list_spc_ns=`grep '	0	' test.dat | cut -f1 | uniq | tr "\n" ' '`
list_spc_ss=`grep '	1	' test.dat | cut -f1 | uniq | tr "\n" ' '`

list_comma_ns=`grep '	0	' test.dat | cut -f1 | uniq | tr "\n" ','`
list_comma_ss=`grep '	1	' test.dat | cut -f1 | uniq | tr "\n" ','`

list_spc=$"$list_spc_ns\\\n$list_spc_ss"

echo -e "$list_spc_ns\n$list_spc_ss"
echo -e "$list_spc_ns\n$list_spc_ss" > ./test_ids_s${split}.dat
echo -e "$list_spc_ns\n$list_spc_ss" > ./test_ids.dat
sed "s/ /\n/g" test_ids.dat | sed -r '/^\s*$/d' > ./test_ids.lst
echo "Written to test_ids_s${split}.dat"

echo -e "$list_comma_ns\n$list_comma_ss"
echo -e "$list_comma_ns\n$list_comma_ss" > ./test_ids_s${split}.inc
echo -e "$list_comma_ns\n$list_comma_ss" > ./test_ids.inc
echo -e "valid_ns = {$list_comma_ns}\nvalid_ss = {$list_comma_ss}" > ./test_ids.py
echo "Written to ./test_ids_s${split}.inc and test_ids.py"

fi


