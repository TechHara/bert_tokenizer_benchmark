set -e

file1=$1
file2=$2

tr ' ' '\n' < $file1 > .flattened1.tmp
tr ' ' '\n' < $file2 > .flattened2.tmp
diff_tokens=$(diff -y --suppress-common-lines .flattened[12].tmp | wc -l | awk '{print $1}')
n1=$(wc -l .flattened1.tmp | awk '{print $1}')
n2=$(wc -l .flattened2.tmp | awk '{print $1}')
max_tokens=`echo "max=$n1; if($n1<$n2) {max=$n2}; max" | bc`

parity=`echo "scale=5; 100 - $diff_tokens / $max_tokens * 100.0" | bc`
echo "parity: $parity% tokens"

rm .flattened[12].tmp