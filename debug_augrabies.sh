module load plink bcftools vcftools king
source /share/hennlab/progs/miniconda3/etc/profile.d/conda.sh
conda activate ped-sim

wdir="/share/hennlab/data/genomes/CAAPA_freeze2_PHASED_common_rare/ponderosa"
#python3 /share/hennlab/data/genomes/CAAPA_freeze2_PHASED_common_rare/ponderosa/simulations_simplified_2/Ponderosa.py \
#python3 Ponderosa.py \
#    --ibd ${wdir}/batwa/chr1.batwa.snps.h3a.maf01.hg19.phasedIBD.txt \
#    --fam ${wdir}/batwa/allchr.batwa.snps.h3a.maf01.hg19.fam \
#    --map ${wdir}/interpolated_maps.caapa_h3a_maf01/newchr1.snps.h3a.maf01.hg19.map \
#    --training ${wdir}/batwa/Simulations/ponderosa_simulations/degree_classifier_pop1.pkl \
#    --output ${wdir}/PonderosaBeta/del_batwa.PONDEROSAresults_2 \
#    --min_p 0.8 \
#    --king /share/hennlab/data/genomes/CAAPA_freeze2_PHASED_common_rare/ponderosa/batwa/allchr.batwa.snps.h3a.maf01.hg19.ibdseg.seg \
#    --debug

python3 ${wdir}/PonderosaBeta/Ponderosa.py \
    --ibd ${wdir}/beja/chr1.beja.snps.h3a.maf01.hg19.phasedIBD.txt \
    --king ${wdir}/beja/allchr.beja.snps.h3a.maf01.hg19.ibdseg.seg \
    --fam ${wdir}/beja/allchr.beja.snps.h3a.maf01.hg19.fam \
    --map ${wdir}/beja/newchr1.beja.snps.h3a.maf01.hg19.map \
    --training ${wdir}/beja/Simulations/ponderosa_simulations/degree_classifier_pop1.pkl \
    --output ${wdir}/PonderosaBeta/del_beja.PONDEROSAresults_2 \
    --min_p 0.8 
