pedsim=${1}
simmap=${2}
intf=${3}
pop=${4}
rel=${5}
iter=${6}
ibdcaller=${7}

$pedsim -i ${pop}_unrelateds.vcf -d ${rel}.def -m $simmap --intf $intf --keep_phase --fam -o ${rel}${iter}

cd ..
pwd
bash ${ibdcaller}.sh $pop $rel $iter

