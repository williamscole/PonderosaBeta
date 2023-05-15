pedsim=${1}
simmap=${2}
intf=${3}
pop=${4}
rel=${5}
iter=${6}
ibdcaller=${7}

$pedsim -i ${pop}_unrelateds.vcf -d ${rel}.def -m $simmap --intf $intf --keep_phase --fam -o ${rel}${iter}

cd ..

bash ${ibdcaller}.sh $pop $rel $iter

python3 pedigree_tools.py ${pop}/${pop}_chr1_${rel}${iter}.txt $pop $rel $iter rename_pop_ibd

#if [[ -f ${rel}_${pop}_sim${iter}_segments.f ]]
#then

#cd ${pop}

#rm ${rel}${iter}_chr*.vcf ${rel}${iter}.seg ${rel}${iter}.vcf ${pop}_unrelateds.vcf

#fi
