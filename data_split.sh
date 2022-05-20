
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

cp -r ../s5c_new/data/eval2000_hires ../s5c_new/data/eval2000_hires_spk
mv ../s5c_new/data/eval2000_hires_spk/feats.scp ../s5c_new/data/eval2000_hires_spk/feats_ori.scp
feat-to-len scp:../s5c_new/data/eval2000_hires_spk/feats_ori.scp ark,t:- | perl /mnt/shareEx/xxr/SWBD/scripts/gen0pdf.pl 1 > ../s5c_new/data/eval2000_hires_spk/align1.pdf
perl /mnt/shareEx/xxr/SWBD/scripts/segment2id.pl ../s5c_new/data/eval2000_hires_spk/utt2spk ../s5c_new/data/eval2000_hires_spk/align1.pdf ../s5c_new/data/eval2000_hires_spk/num_spk > ../s5c_new/data/eval2000_hires_spk/spk
perl /mnt/shareEx/xxr/SWBD/scripts/pdf2ark_simple.pl ../s5c_new/data/eval2000_hires_spk/spk > ../s5c_new/data/eval2000_hires_spk/spk.ark
analyze-counts --binary=false ark:../s5c_new/data/eval2000_hires_spk/spk ../s5c_new/data/eval2000_hires_spk/spk_count

paste-feats scp:../s5c_new/data/eval2000_hires_spk/feats_ori.scp ark:../s5c_new/data/eval2000_hires_spk/spk.ark ark,scp:../s5c_new/data/eval2000_hires_spk/feats.ark,../s5c_new/data/eval2000_hires_spk/feats.scp
steps/compute_cmvn_stats.sh ../s5c_new/data/eval2000_hires_spk



. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

for N in {5,10,20,40}; do

cp -r ../s5c_new/data/eval2000_hires ../s5c_new/data/eval2000_hires_spk_sub${N}
mv ../s5c_new/data/eval2000_hires_spk_sub${N}/feats.scp ../s5c_new/data/eval2000_hires_spk_sub${N}/feats_ori.scp
perl /mnt/shareEx/xxr/SWBD/scripts/utt2spk_split_everyN.pl ../s5c_new/data/eval2000_hires_spk/utt2spk $N | grep "_sub1$" > ../s5c_new/data/eval2000_hires_spk_sub${N}/utt2spk
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/eval2000_hires_spk/align1.pdf ../s5c_new/data/eval2000_hires_spk_sub${N}/utt2spk > ../s5c_new/data/eval2000_hires_spk_sub${N}/align1.pdf
perl /mnt/shareEx/xxr/SWBD/scripts/segment2id.pl ../s5c_new/data/eval2000_hires_spk_sub${N}/utt2spk ../s5c_new/data/eval2000_hires_spk_sub${N}/align1.pdf ../s5c_new/data/eval2000_hires_spk_sub${N}/num_spk > ../s5c_new/data/eval2000_hires_spk_sub${N}/spk
perl /mnt/shareEx/xxr/SWBD/scripts/pdf2ark_simple.pl ../s5c_new/data/eval2000_hires_spk_sub${N}/spk > ../s5c_new/data/eval2000_hires_spk_sub${N}/spk.ark
num=`cat ../s5c_new/data/eval2000_hires_spk_sub${N}/num_spk`
analyze-counts --binary=false ark:../s5c_new/data/eval2000_hires_spk_sub${N}/spk ../s5c_new/data/eval2000_hires_spk_sub${N}/spk_count

paste-feats scp:../s5c_new/data/eval2000_hires_spk_sub${N}/feats_ori.scp ark:../s5c_new/data/eval2000_hires_spk_sub${N}/spk.ark ark,scp:../s5c_new/data/eval2000_hires_spk_sub${N}/feats.ark,../s5c_new/data/eval2000_hires_spk_sub${N}/feats.scp
steps/compute_cmvn_stats.sh ../s5c_new/data/eval2000_hires_spk_sub${N}
mv ../s5c_new/data/eval2000_hires_spk_sub${N}/text ../s5c_new/data/eval2000_hires_spk_sub${N}/text_all
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/eval2000_hires_spk_sub${N}/text_all ../s5c_new/data/eval2000_hires_spk_sub${N}/feats.scp > ../s5c_new/data/eval2000_hires_spk_sub${N}/text
perl utils/../s5c_new/data/get_utt2dur.sh ../s5c_new/data/eval2000_hires_spk_sub${N}
mv ../s5c_new/data/eval2000_hires_spk_sub${N}/utt2dur ../s5c_new/data/eval2000_hires_spk_sub${N}/utt2dur_all
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/eval2000_hires_spk_sub${N}/utt2dur_all ../s5c_new/data/eval2000_hires_spk_sub${N}/feats.scp > ../s5c_new/data/eval2000_hires_spk_sub${N}/utt2dur

done




. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

cp -r ../s5c_new/data/eval2000_sp_hires ../s5c_new/data/eval2000_sp_hires_spk
mv ../s5c_new/data/eval2000_sp_hires_spk/feats.scp ../s5c_new/data/eval2000_sp_hires_spk/feats_ori.scp
feat-to-len scp:../s5c_new/data/eval2000_sp_hires_spk/feats_ori.scp ark,t:- | perl /mnt/shareEx/xxr/SWBD/scripts/gen0pdf.pl 1 > ../s5c_new/data/eval2000_sp_hires_spk/align1.pdf
cat ../s5c_new/data/eval2000_sp_hires_spk/utt2spk | sed "s/ sp.\..-/ /g" > ../s5c_new/data/eval2000_sp_hires_spk/utt2spk_all
perl /mnt/shareEx/xxr/SWBD/scripts/segment2id.pl ../s5c_new/data/eval2000_sp_hires_spk/utt2spk_all ../s5c_new/data/eval2000_sp_hires_spk/align1.pdf ../s5c_new/data/eval2000_sp_hires_spk/num_spk > ../s5c_new/data/eval2000_sp_hires_spk/spk
perl /mnt/shareEx/xxr/SWBD/scripts/pdf2ark_simple.pl ../s5c_new/data/eval2000_sp_hires_spk/spk > ../s5c_new/data/eval2000_sp_hires_spk/spk.ark
analyze-counts --binary=false ark:../s5c_new/data/eval2000_sp_hires_spk/spk ../s5c_new/data/eval2000_sp_hires_spk/spk_count

paste-feats scp:../s5c_new/data/eval2000_sp_hires_spk/feats_ori.scp ark:../s5c_new/data/eval2000_sp_hires_spk/spk.ark ark,scp:../s5c_new/data/eval2000_sp_hires_spk/feats.ark,../s5c_new/data/eval2000_sp_hires_spk/feats.scp
steps/compute_cmvn_stats.sh ../s5c_new/data/eval2000_sp_hires_spk


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

for N in {5,10,20,40}; do

cp -r ../s5c_new/data/eval2000_sp_hires ../s5c_new/data/eval2000_sp_hires_spk_sub${N}
mv ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/feats.scp ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/feats_ori.scp
perl /mnt/shareEx/xxr/SWBD/scripts/utt2spk_split_everyN.pl ../s5c_new/data/eval2000_sp_hires_spk/utt2spk $N | grep "_sub1$" | sed "s/ sp.\..-/ /g" > ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/utt2spk_all
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/eval2000_sp_hires_spk/align1.pdf ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/utt2spk_all > ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/align1.pdf
perl /mnt/shareEx/xxr/SWBD/scripts/segment2id.pl ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/utt2spk_all ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/align1.pdf ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/num_spk > ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/spk
perl /mnt/shareEx/xxr/SWBD/scripts/pdf2ark_simple.pl ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/spk > ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/spk.ark
num=`cat ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/num_spk`
analyze-counts --binary=false ark:../s5c_new/data/eval2000_sp_hires_spk_sub${N}/spk ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/spk_count

paste-feats scp:../s5c_new/data/eval2000_sp_hires_spk_sub${N}/feats_ori.scp ark:../s5c_new/data/eval2000_sp_hires_spk_sub${N}/spk.ark ark,scp:../s5c_new/data/eval2000_sp_hires_spk_sub${N}/feats.ark,../s5c_new/data/eval2000_sp_hires_spk_sub${N}/feats.scp
steps/compute_cmvn_stats.sh ../s5c_new/data/eval2000_sp_hires_spk_sub${N}
mv ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/text ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/text_all
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/text_all ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/feats.scp > ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/text
perl utils/../s5c_new/data/get_utt2dur.sh ../s5c_new/data/eval2000_sp_hires_spk_sub${N}
mv ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/utt2dur ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/utt2dur_all
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/utt2dur_all ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/feats.scp > ../s5c_new/data/eval2000_sp_hires_spk_sub${N}/utt2dur

done





. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

cp -r ../s5c_new/data/rt03_hires ../s5c_new/data/rt03_hires_spk
mv ../s5c_new/data/rt03_hires_spk/feats.scp ../s5c_new/data/rt03_hires_spk/feats_ori.scp
feat-to-len scp:../s5c_new/data/rt03_hires_spk/feats_ori.scp ark,t:- | perl /mnt/shareEx/xxr/SWBD/scripts/gen0pdf.pl 1 > ../s5c_new/data/rt03_hires_spk/align1.pdf
perl /mnt/shareEx/xxr/SWBD/scripts/segment2id.pl ../s5c_new/data/rt03_hires_spk/utt2spk ../s5c_new/data/rt03_hires_spk/align1.pdf ../s5c_new/data/rt03_hires_spk/num_spk > ../s5c_new/data/rt03_hires_spk/spk
perl /mnt/shareEx/xxr/SWBD/scripts/pdf2ark_simple.pl ../s5c_new/data/rt03_hires_spk/spk > ../s5c_new/data/rt03_hires_spk/spk.ark
analyze-counts --binary=false ark:../s5c_new/data/rt03_hires_spk/spk ../s5c_new/data/rt03_hires_spk/spk_count

paste-feats scp:../s5c_new/data/rt03_hires_spk/feats_ori.scp ark:../s5c_new/data/rt03_hires_spk/spk.ark ark,scp:../s5c_new/data/rt03_hires_spk/feats.ark,../s5c_new/data/rt03_hires_spk/feats.scp
steps/compute_cmvn_stats.sh ../s5c_new/data/rt03_hires_spk



. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

for N in {5,10,20,40}; do

cp -r ../s5c_new/data/rt03_hires ../s5c_new/data/rt03_hires_spk_sub${N}
mv ../s5c_new/data/rt03_hires_spk_sub${N}/feats.scp ../s5c_new/data/rt03_hires_spk_sub${N}/feats_ori.scp
perl /mnt/shareEx/xxr/SWBD/scripts/utt2spk_split_everyN.pl ../s5c_new/data/rt03_hires_spk/utt2spk $N | grep "_sub1$" > ../s5c_new/data/rt03_hires_spk_sub${N}/utt2spk
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/rt03_hires_spk/align1.pdf ../s5c_new/data/rt03_hires_spk_sub${N}/utt2spk > ../s5c_new/data/rt03_hires_spk_sub${N}/align1.pdf
perl /mnt/shareEx/xxr/SWBD/scripts/segment2id.pl ../s5c_new/data/rt03_hires_spk_sub${N}/utt2spk ../s5c_new/data/rt03_hires_spk_sub${N}/align1.pdf ../s5c_new/data/rt03_hires_spk_sub${N}/num_spk > ../s5c_new/data/rt03_hires_spk_sub${N}/spk
perl /mnt/shareEx/xxr/SWBD/scripts/pdf2ark_simple.pl ../s5c_new/data/rt03_hires_spk_sub${N}/spk > ../s5c_new/data/rt03_hires_spk_sub${N}/spk.ark
num=`cat ../s5c_new/data/rt03_hires_spk_sub${N}/num_spk`
analyze-counts --binary=false ark:../s5c_new/data/rt03_hires_spk_sub${N}/spk ../s5c_new/data/rt03_hires_spk_sub${N}/spk_count

paste-feats scp:../s5c_new/data/rt03_hires_spk_sub${N}/feats_ori.scp ark:../s5c_new/data/rt03_hires_spk_sub${N}/spk.ark ark,scp:../s5c_new/data/rt03_hires_spk_sub${N}/feats.ark,../s5c_new/data/rt03_hires_spk_sub${N}/feats.scp
steps/compute_cmvn_stats.sh ../s5c_new/data/rt03_hires_spk_sub${N}
mv ../s5c_new/data/rt03_hires_spk_sub${N}/text ../s5c_new/data/rt03_hires_spk_sub${N}/text_all
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/rt03_hires_spk_sub${N}/text_all ../s5c_new/data/rt03_hires_spk_sub${N}/feats.scp > ../s5c_new/data/rt03_hires_spk_sub${N}/text
perl utils/../s5c_new/data/get_utt2dur.sh ../s5c_new/data/rt03_hires_spk_sub${N}
mv ../s5c_new/data/rt03_hires_spk_sub${N}/utt2dur ../s5c_new/data/rt03_hires_spk_sub${N}/utt2dur_all
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/rt03_hires_spk_sub${N}/utt2dur_all ../s5c_new/data/rt03_hires_spk_sub${N}/feats.scp > ../s5c_new/data/rt03_hires_spk_sub${N}/utt2dur

done




. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

cp -r ../s5c_new/data/rt03_sp_hires ../s5c_new/data/rt03_sp_hires_spk
mv ../s5c_new/data/rt03_sp_hires_spk/feats.scp ../s5c_new/data/rt03_sp_hires_spk/feats_ori.scp
feat-to-len scp:../s5c_new/data/rt03_sp_hires_spk/feats_ori.scp ark,t:- | perl /mnt/shareEx/xxr/SWBD/scripts/gen0pdf.pl 1 > ../s5c_new/data/rt03_sp_hires_spk/align1.pdf
cat ../s5c_new/data/rt03_sp_hires_spk/utt2spk | sed "s/ sp.\..-/ /g" > ../s5c_new/data/rt03_sp_hires_spk/utt2spk_all
perl /mnt/shareEx/xxr/SWBD/scripts/segment2id.pl ../s5c_new/data/rt03_sp_hires_spk/utt2spk_all ../s5c_new/data/rt03_sp_hires_spk/align1.pdf ../s5c_new/data/rt03_sp_hires_spk/num_spk > ../s5c_new/data/rt03_sp_hires_spk/spk
perl /mnt/shareEx/xxr/SWBD/scripts/pdf2ark_simple.pl ../s5c_new/data/rt03_sp_hires_spk/spk > ../s5c_new/data/rt03_sp_hires_spk/spk.ark
analyze-counts --binary=false ark:../s5c_new/data/rt03_sp_hires_spk/spk ../s5c_new/data/rt03_sp_hires_spk/spk_count

paste-feats scp:../s5c_new/data/rt03_sp_hires_spk/feats_ori.scp ark:../s5c_new/data/rt03_sp_hires_spk/spk.ark ark,scp:../s5c_new/data/rt03_sp_hires_spk/feats.ark,../s5c_new/data/rt03_sp_hires_spk/feats.scp
steps/compute_cmvn_stats.sh ../s5c_new/data/rt03_sp_hires_spk


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

for N in {5,10,20,40}; do

cp -r ../s5c_new/data/rt03_sp_hires ../s5c_new/data/rt03_sp_hires_spk_sub${N}
mv ../s5c_new/data/rt03_sp_hires_spk_sub${N}/feats.scp ../s5c_new/data/rt03_sp_hires_spk_sub${N}/feats_ori.scp
perl /mnt/shareEx/xxr/SWBD/scripts/utt2spk_split_everyN.pl ../s5c_new/data/rt03_sp_hires_spk/utt2spk $N | grep "_sub1$" | sed "s/ sp.\..-/ /g" > ../s5c_new/data/rt03_sp_hires_spk_sub${N}/utt2spk_all
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/rt03_sp_hires_spk/align1.pdf ../s5c_new/data/rt03_sp_hires_spk_sub${N}/utt2spk_all > ../s5c_new/data/rt03_sp_hires_spk_sub${N}/align1.pdf
perl /mnt/shareEx/xxr/SWBD/scripts/segment2id.pl ../s5c_new/data/rt03_sp_hires_spk_sub${N}/utt2spk_all ../s5c_new/data/rt03_sp_hires_spk_sub${N}/align1.pdf ../s5c_new/data/rt03_sp_hires_spk_sub${N}/num_spk > ../s5c_new/data/rt03_sp_hires_spk_sub${N}/spk
perl /mnt/shareEx/xxr/SWBD/scripts/pdf2ark_simple.pl ../s5c_new/data/rt03_sp_hires_spk_sub${N}/spk > ../s5c_new/data/rt03_sp_hires_spk_sub${N}/spk.ark
num=`cat ../s5c_new/data/rt03_sp_hires_spk_sub${N}/num_spk`
analyze-counts --binary=false ark:../s5c_new/data/rt03_sp_hires_spk_sub${N}/spk ../s5c_new/data/rt03_sp_hires_spk_sub${N}/spk_count

paste-feats scp:../s5c_new/data/rt03_sp_hires_spk_sub${N}/feats_ori.scp ark:../s5c_new/data/rt03_sp_hires_spk_sub${N}/spk.ark ark,scp:../s5c_new/data/rt03_sp_hires_spk_sub${N}/feats.ark,../s5c_new/data/rt03_sp_hires_spk_sub${N}/feats.scp
steps/compute_cmvn_stats.sh ../s5c_new/data/rt03_sp_hires_spk_sub${N}
mv ../s5c_new/data/rt03_sp_hires_spk_sub${N}/text ../s5c_new/data/rt03_sp_hires_spk_sub${N}/text_all
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/rt03_sp_hires_spk_sub${N}/text_all ../s5c_new/data/rt03_sp_hires_spk_sub${N}/feats.scp > ../s5c_new/data/rt03_sp_hires_spk_sub${N}/text
perl utils/../s5c_new/data/get_utt2dur.sh ../s5c_new/data/rt03_sp_hires_spk_sub${N}
mv ../s5c_new/data/rt03_sp_hires_spk_sub${N}/utt2dur ../s5c_new/data/rt03_sp_hires_spk_sub${N}/utt2dur_all
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/rt03_sp_hires_spk_sub${N}/utt2dur_all ../s5c_new/data/rt03_sp_hires_spk_sub${N}/feats.scp > ../s5c_new/data/rt03_sp_hires_spk_sub${N}/utt2dur

done







. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

cp -r ../s5c_new/data/eval2000_fbk_40 ../s5c_new/data/eval2000_fbk_40_spk
mv ../s5c_new/data/eval2000_fbk_40_spk/feats.scp ../s5c_new/data/eval2000_fbk_40_spk/feats_ori.scp
feat-to-len scp:../s5c_new/data/eval2000_fbk_40_spk/feats_ori.scp ark,t:- | perl /mnt/shareEx/xxr/SWBD/scripts/gen0pdf.pl 1 > ../s5c_new/data/eval2000_fbk_40_spk/align1.pdf
perl /mnt/shareEx/xxr/SWBD/scripts/segment2id.pl ../s5c_new/data/eval2000_fbk_40_spk/utt2spk ../s5c_new/data/eval2000_fbk_40_spk/align1.pdf ../s5c_new/data/eval2000_fbk_40_spk/num_spk > ../s5c_new/data/eval2000_fbk_40_spk/spk
perl /mnt/shareEx/xxr/SWBD/scripts/pdf2ark_simple.pl ../s5c_new/data/eval2000_fbk_40_spk/spk > ../s5c_new/data/eval2000_fbk_40_spk/spk.ark
analyze-counts --binary=false ark:../s5c_new/data/eval2000_fbk_40_spk/spk ../s5c_new/data/eval2000_fbk_40_spk/spk_count

paste-feats scp:../s5c_new/data/eval2000_fbk_40_spk/feats_ori.scp ark:../s5c_new/data/eval2000_fbk_40_spk/spk.ark ark,scp:../s5c_new/data/eval2000_fbk_40_spk/feats.ark,../s5c_new/data/eval2000_fbk_40_spk/feats.scp
steps/compute_cmvn_stats.sh ../s5c_new/data/eval2000_fbk_40_spk



. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

for N in {5,10,20,40}; do

cp -r ../s5c_new/data/eval2000_fbk_40 ../s5c_new/data/eval2000_fbk_40_spk_sub${N}
mv ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/feats.scp ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/feats_ori.scp
perl /mnt/shareEx/xxr/SWBD/scripts/utt2spk_split_everyN.pl ../s5c_new/data/eval2000_fbk_40_spk/utt2spk $N | grep "_sub1$" > ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/utt2spk
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/eval2000_fbk_40_spk/align1.pdf ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/utt2spk > ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/align1.pdf
perl /mnt/shareEx/xxr/SWBD/scripts/segment2id.pl ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/utt2spk ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/align1.pdf ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/num_spk > ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/spk
perl /mnt/shareEx/xxr/SWBD/scripts/pdf2ark_simple.pl ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/spk > ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/spk.ark
num=`cat ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/num_spk`
analyze-counts --binary=false ark:../s5c_new/data/eval2000_fbk_40_spk_sub${N}/spk ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/spk_count

paste-feats scp:../s5c_new/data/eval2000_fbk_40_spk_sub${N}/feats_ori.scp ark:../s5c_new/data/eval2000_fbk_40_spk_sub${N}/spk.ark ark,scp:../s5c_new/data/eval2000_fbk_40_spk_sub${N}/feats.ark,../s5c_new/data/eval2000_fbk_40_spk_sub${N}/feats.scp
steps/compute_cmvn_stats.sh ../s5c_new/data/eval2000_fbk_40_spk_sub${N}
mv ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/text ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/text_all
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/text_all ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/feats.scp > ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/text
perl utils/../s5c_new/data/get_utt2dur.sh ../s5c_new/data/eval2000_fbk_40_spk_sub${N}
mv ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/utt2dur ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/utt2dur_all
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/utt2dur_all ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/feats.scp > ../s5c_new/data/eval2000_fbk_40_spk_sub${N}/utt2dur

done




. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

cp -r ../s5c_new/data/eval2000_fbk_sp_40 ../s5c_new/data/eval2000_fbk_sp_40_spk
mv ../s5c_new/data/eval2000_fbk_sp_40_spk/feats.scp ../s5c_new/data/eval2000_fbk_sp_40_spk/feats_ori.scp
feat-to-len scp:../s5c_new/data/eval2000_fbk_sp_40_spk/feats_ori.scp ark,t:- | perl /mnt/shareEx/xxr/SWBD/scripts/gen0pdf.pl 1 > ../s5c_new/data/eval2000_fbk_sp_40_spk/align1.pdf
cat ../s5c_new/data/eval2000_fbk_sp_40_spk/utt2spk | sed "s/ sp.\..-/ /g" > ../s5c_new/data/eval2000_fbk_sp_40_spk/utt2spk_all
perl /mnt/shareEx/xxr/SWBD/scripts/segment2id.pl ../s5c_new/data/eval2000_fbk_sp_40_spk/utt2spk_all ../s5c_new/data/eval2000_fbk_sp_40_spk/align1.pdf ../s5c_new/data/eval2000_fbk_sp_40_spk/num_spk > ../s5c_new/data/eval2000_fbk_sp_40_spk/spk
perl /mnt/shareEx/xxr/SWBD/scripts/pdf2ark_simple.pl ../s5c_new/data/eval2000_fbk_sp_40_spk/spk > ../s5c_new/data/eval2000_fbk_sp_40_spk/spk.ark
analyze-counts --binary=false ark:../s5c_new/data/eval2000_fbk_sp_40_spk/spk ../s5c_new/data/eval2000_fbk_sp_40_spk/spk_count

paste-feats scp:../s5c_new/data/eval2000_fbk_sp_40_spk/feats_ori.scp ark:../s5c_new/data/eval2000_fbk_sp_40_spk/spk.ark ark,scp:../s5c_new/data/eval2000_fbk_sp_40_spk/feats.ark,../s5c_new/data/eval2000_fbk_sp_40_spk/feats.scp
steps/compute_cmvn_stats.sh ../s5c_new/data/eval2000_fbk_sp_40_spk


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

for N in {5,10,20,40}; do

cp -r ../s5c_new/data/eval2000_fbk_sp_40 ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}
mv ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/feats.scp ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/feats_ori.scp
perl /mnt/shareEx/xxr/SWBD/scripts/utt2spk_split_everyN.pl ../s5c_new/data/eval2000_fbk_sp_40_spk/utt2spk $N | grep "_sub1$" | sed "s/ sp.\..-/ /g" > ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/utt2spk_all
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/eval2000_fbk_sp_40_spk/align1.pdf ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/utt2spk_all > ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/align1.pdf
perl /mnt/shareEx/xxr/SWBD/scripts/segment2id.pl ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/utt2spk_all ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/align1.pdf ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/num_spk > ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/spk
perl /mnt/shareEx/xxr/SWBD/scripts/pdf2ark_simple.pl ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/spk > ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/spk.ark
num=`cat ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/num_spk`
analyze-counts --binary=false ark:../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/spk ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/spk_count

paste-feats scp:../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/feats_ori.scp ark:../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/spk.ark ark,scp:../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/feats.ark,../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/feats.scp
steps/compute_cmvn_stats.sh ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}
mv ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/text ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/text_all
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/text_all ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/feats.scp > ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/text
perl utils/../s5c_new/data/get_utt2dur.sh ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}
mv ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/utt2dur ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/utt2dur_all
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/utt2dur_all ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/feats.scp > ../s5c_new/data/eval2000_fbk_sp_40_spk_sub${N}/utt2dur

done





. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

cp -r ../s5c_new/data/rt03_fbk_40 ../s5c_new/data/rt03_fbk_40_spk
mv ../s5c_new/data/rt03_fbk_40_spk/feats.scp ../s5c_new/data/rt03_fbk_40_spk/feats_ori.scp
feat-to-len scp:../s5c_new/data/rt03_fbk_40_spk/feats_ori.scp ark,t:- | perl /mnt/shareEx/xxr/SWBD/scripts/gen0pdf.pl 1 > ../s5c_new/data/rt03_fbk_40_spk/align1.pdf
perl /mnt/shareEx/xxr/SWBD/scripts/segment2id.pl ../s5c_new/data/rt03_fbk_40_spk/utt2spk ../s5c_new/data/rt03_fbk_40_spk/align1.pdf ../s5c_new/data/rt03_fbk_40_spk/num_spk > ../s5c_new/data/rt03_fbk_40_spk/spk
perl /mnt/shareEx/xxr/SWBD/scripts/pdf2ark_simple.pl ../s5c_new/data/rt03_fbk_40_spk/spk > ../s5c_new/data/rt03_fbk_40_spk/spk.ark
analyze-counts --binary=false ark:../s5c_new/data/rt03_fbk_40_spk/spk ../s5c_new/data/rt03_fbk_40_spk/spk_count

paste-feats scp:../s5c_new/data/rt03_fbk_40_spk/feats_ori.scp ark:../s5c_new/data/rt03_fbk_40_spk/spk.ark ark,scp:../s5c_new/data/rt03_fbk_40_spk/feats.ark,../s5c_new/data/rt03_fbk_40_spk/feats.scp
steps/compute_cmvn_stats.sh ../s5c_new/data/rt03_fbk_40_spk



. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

for N in {5,10,20,40}; do

cp -r ../s5c_new/data/rt03_fbk_40 ../s5c_new/data/rt03_fbk_40_spk_sub${N}
mv ../s5c_new/data/rt03_fbk_40_spk_sub${N}/feats.scp ../s5c_new/data/rt03_fbk_40_spk_sub${N}/feats_ori.scp
perl /mnt/shareEx/xxr/SWBD/scripts/utt2spk_split_everyN.pl ../s5c_new/data/rt03_fbk_40_spk/utt2spk $N | grep "_sub1$" > ../s5c_new/data/rt03_fbk_40_spk_sub${N}/utt2spk
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/rt03_fbk_40_spk/align1.pdf ../s5c_new/data/rt03_fbk_40_spk_sub${N}/utt2spk > ../s5c_new/data/rt03_fbk_40_spk_sub${N}/align1.pdf
perl /mnt/shareEx/xxr/SWBD/scripts/segment2id.pl ../s5c_new/data/rt03_fbk_40_spk_sub${N}/utt2spk ../s5c_new/data/rt03_fbk_40_spk_sub${N}/align1.pdf ../s5c_new/data/rt03_fbk_40_spk_sub${N}/num_spk > ../s5c_new/data/rt03_fbk_40_spk_sub${N}/spk
perl /mnt/shareEx/xxr/SWBD/scripts/pdf2ark_simple.pl ../s5c_new/data/rt03_fbk_40_spk_sub${N}/spk > ../s5c_new/data/rt03_fbk_40_spk_sub${N}/spk.ark
num=`cat ../s5c_new/data/rt03_fbk_40_spk_sub${N}/num_spk`
analyze-counts --binary=false ark:../s5c_new/data/rt03_fbk_40_spk_sub${N}/spk ../s5c_new/data/rt03_fbk_40_spk_sub${N}/spk_count

paste-feats scp:../s5c_new/data/rt03_fbk_40_spk_sub${N}/feats_ori.scp ark:../s5c_new/data/rt03_fbk_40_spk_sub${N}/spk.ark ark,scp:../s5c_new/data/rt03_fbk_40_spk_sub${N}/feats.ark,../s5c_new/data/rt03_fbk_40_spk_sub${N}/feats.scp
steps/compute_cmvn_stats.sh ../s5c_new/data/rt03_fbk_40_spk_sub${N}
mv ../s5c_new/data/rt03_fbk_40_spk_sub${N}/text ../s5c_new/data/rt03_fbk_40_spk_sub${N}/text_all
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/rt03_fbk_40_spk_sub${N}/text_all ../s5c_new/data/rt03_fbk_40_spk_sub${N}/feats.scp > ../s5c_new/data/rt03_fbk_40_spk_sub${N}/text
perl utils/../s5c_new/data/get_utt2dur.sh ../s5c_new/data/rt03_fbk_40_spk_sub${N}
mv ../s5c_new/data/rt03_fbk_40_spk_sub${N}/utt2dur ../s5c_new/data/rt03_fbk_40_spk_sub${N}/utt2dur_all
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/rt03_fbk_40_spk_sub${N}/utt2dur_all ../s5c_new/data/rt03_fbk_40_spk_sub${N}/feats.scp > ../s5c_new/data/rt03_fbk_40_spk_sub${N}/utt2dur

done




. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

cp -r ../s5c_new/data/rt03_fbk_sp_40 ../s5c_new/data/rt03_fbk_sp_40_spk
mv ../s5c_new/data/rt03_fbk_sp_40_spk/feats.scp ../s5c_new/data/rt03_fbk_sp_40_spk/feats_ori.scp
feat-to-len scp:../s5c_new/data/rt03_fbk_sp_40_spk/feats_ori.scp ark,t:- | perl /mnt/shareEx/xxr/SWBD/scripts/gen0pdf.pl 1 > ../s5c_new/data/rt03_fbk_sp_40_spk/align1.pdf
cat ../s5c_new/data/rt03_fbk_sp_40_spk/utt2spk | sed "s/ sp.\..-/ /g" > ../s5c_new/data/rt03_fbk_sp_40_spk/utt2spk_all
perl /mnt/shareEx/xxr/SWBD/scripts/segment2id.pl ../s5c_new/data/rt03_fbk_sp_40_spk/utt2spk_all ../s5c_new/data/rt03_fbk_sp_40_spk/align1.pdf ../s5c_new/data/rt03_fbk_sp_40_spk/num_spk > ../s5c_new/data/rt03_fbk_sp_40_spk/spk
perl /mnt/shareEx/xxr/SWBD/scripts/pdf2ark_simple.pl ../s5c_new/data/rt03_fbk_sp_40_spk/spk > ../s5c_new/data/rt03_fbk_sp_40_spk/spk.ark
analyze-counts --binary=false ark:../s5c_new/data/rt03_fbk_sp_40_spk/spk ../s5c_new/data/rt03_fbk_sp_40_spk/spk_count

paste-feats scp:../s5c_new/data/rt03_fbk_sp_40_spk/feats_ori.scp ark:../s5c_new/data/rt03_fbk_sp_40_spk/spk.ark ark,scp:../s5c_new/data/rt03_fbk_sp_40_spk/feats.ark,../s5c_new/data/rt03_fbk_sp_40_spk/feats.scp
steps/compute_cmvn_stats.sh ../s5c_new/data/rt03_fbk_sp_40_spk


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

for N in {5,10,20,40}; do

cp -r ../s5c_new/data/rt03_fbk_sp_40 ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}
mv ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/feats.scp ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/feats_ori.scp
perl /mnt/shareEx/xxr/SWBD/scripts/utt2spk_split_everyN.pl ../s5c_new/data/rt03_fbk_sp_40_spk/utt2spk $N | grep "_sub1$" | sed "s/ sp.\..-/ /g" > ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/utt2spk_all
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/rt03_fbk_sp_40_spk/align1.pdf ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/utt2spk_all > ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/align1.pdf
perl /mnt/shareEx/xxr/SWBD/scripts/segment2id.pl ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/utt2spk_all ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/align1.pdf ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/num_spk > ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/spk
perl /mnt/shareEx/xxr/SWBD/scripts/pdf2ark_simple.pl ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/spk > ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/spk.ark
num=`cat ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/num_spk`
analyze-counts --binary=false ark:../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/spk ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/spk_count

paste-feats scp:../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/feats_ori.scp ark:../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/spk.ark ark,scp:../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/feats.ark,../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/feats.scp
steps/compute_cmvn_stats.sh ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}
mv ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/text ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/text_all
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/text_all ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/feats.scp > ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/text
perl utils/../s5c_new/data/get_utt2dur.sh ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}
mv ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/utt2dur ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/utt2dur_all
perl /mnt/shareEx/xxr/SWBD/scripts/find_pdf.pl ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/utt2dur_all ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/feats.scp > ../s5c_new/data/rt03_fbk_sp_40_spk_sub${N}/utt2dur

done

