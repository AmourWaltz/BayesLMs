#!/usr/bin/env bash

# This script is very similar to rnnlm/lmrescore_nbest.sh, and it performs N-best
# LM rescoring with Pytorch trained neural LMs.

# Begin configuration section.
N=10
model_type=LSTM # LSTM, GRU or Transformer
embedding_dim=650
hidden_dim=650
nlayers=2
nhead=6
inv_acwt=10
cmd=run.pl
uncertainty=Gaussian
T_bayes_pos=FFN # bayes position, options:none, FFN, MHA, EMB
L_bayes_pos=0 # LSTM Bayesian position: [0: standard | 1: input_gate | 2: forget_gate | 3: cell_gate | 4: output_gate]
L_gauss_pos=0
L_v_pos=00
T_v_pos=0
T_gauss_pos=0
interpolation_flag=0
inter_alpha=0.8
use_phi=false  # This is kind of an obscure option.  If true, we'll remove the old
  # LM weights (times 1-RNN_scale) using a phi (failure) matcher, which is
  # appropriate if the old LM weights were added in this way, e.g. by
  # lmrescore.sh.  Otherwise we'll use normal composition, which is appropriate
  # if the lattices came directly from decoding.  This won't actually make much
  # difference (if any) to WER, it's more so we know we are doing the right thing.
test=false # Activate a testing option.
stage=1 # Stage of this script, for partial reruns.
skip_scoring=false
keep_ali=true
# End configuration section.

echo "$0 $*"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh
. utils/parse_options.sh

echo "$2"
if [ $# != 7 ]; then
   echo "Do language model rescoring of lattices (partially remove old LM, add new LM)"
   echo "This version applies an neural LM and mixes it with the n-gram LM scores"
   echo "previously in the lattices, controlled by the first parameter (nnlm-weight)"
   echo ""
   echo "Usage: $0 [options] <nn-weight> <old-lang-dir> <nn-model-dir> vocab <data-dir> <input-decode-dir> <output-decode-dir>"
   echo "Main options:"
   echo "  --inv-acwt <inv-acwt>          # default 12.  e.g. --inv-acwt 17.  Equivalent to LM scale to use."
   echo "                                 # for N-best list generation... note, we'll score at different acwt's"
   echo "  --cmd <run.pl|queue.pl [opts]> # how to run jobs."
   echo "  --phi (true|false)             # Should be set to true if the source lattices were created"
   echo "                                 # by lmrescore.sh, false if they came from decoding."
   echo "  --N <N>                        # Value of N in N-best rescoring (default: 10)"
   exit 1;
fi

nnweight=$1 # weight of a neural network LM
oldlang=$2
nn_model=$3
vocabulary=$4
data=$5
indir=$6
dir=$7

acwt=$(perl -e "print (1.0/$inv_acwt);")

# Figures out if the old LM is G.fst or G.carpa
oldlm=$oldlang/G.fst
if [ -f $oldlang/G.carpa ]; then
  oldlm=$oldlang/G.carpa
elif [ ! -f $oldlm ]; then
  echo "$0: expecting either $oldlang/G.fst or $oldlang/G.carpa to exist" &&\
    exit 1;
fi

for f in $nn_model $vocabulary $indir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist." && exit 1;
done

nj=5
#nj=$(cat $indir/num_jobs.new) || exit 1;
#nj=$(cat $indir/num_jobs) || exit 1; # for ami without fisher
mkdir -p $dir;
cp $indir/num_jobs $dir/num_jobs
cp $indir/num_jobs.new $dir/num_jobs.new

adir=$dir/archives

phi=$(grep -w '#0' $oldlang/words.txt | awk '{print $2}')

rm $dir/.error 2>/dev/null
mkdir -p $dir/log

# First convert lattice to N-best.  Be careful because this
# will be quite sensitive to the acoustic scale; this should be close
# to the one we'll finally get the best WERs with.
# Note: the lattice-rmali part here is just because we don't
# need the alignments for what we're doing.
if [ $stage -le 1 ]; then # lat.JOB.new.gz
  echo "$0: converting lattices to N-best lists."
  if $keep_ali; then
    $cmd JOB=1:$nj $dir/log/lat2nbest.JOB.log \
      lattice-to-nbest --acoustic-scale=$acwt --n=$N \
      "ark:gunzip -c $indir/lat.JOB.new.gz|" \
      "ark:|gzip -c >$dir/nbest1.JOB.gz" || exit 1;
  else
    $cmd JOB=1:$nj $dir/log/lat2nbest.JOB.log \
      lattice-to-nbest --acoustic-scale=$acwt --n=$N \
      "ark:gunzip -c $indir/lat.JOB.gz|" ark:- \|  \
      lattice-rmali ark:- "ark:|gzip -c >$dir/nbest1.JOB.gz" || exit 1;
  fi
fi

# next remove part of the old LM probs.
if [ "$oldlm" == "$oldlang/G.fst" ]; then
  if $use_phi; then
    if [ $stage -le 2 ]; then
      echo "$0: removing old LM scores."
      # Use the phi-matcher style of composition.. this is appropriate
      # if the old LM scores were added e.g. by lmrescore.sh, using
      # phi-matcher composition.
      $cmd JOB=1:$nj $dir/log/remove_old.JOB.log \
        lattice-scale --acoustic-scale=-1 --lm-scale=-1 "ark:gunzip -c $dir/nbest1.JOB.gz|" ark:- \| \
        lattice-compose --phi-label=$phi ark:- $oldlm ark:- \| \
        lattice-scale --acoustic-scale=-1 --lm-scale=-1 ark:- "ark:|gzip -c >$dir/nbest2.JOB.gz" \
        || exit 1;
    fi
  else
    if [ $stage -le 2 ]; then
      echo "$0: removing old LM scores."
      # this approach chooses the best path through the old LM FST, while
      # subtracting the old scores.  If the lattices came straight from decoding,
      # this is what we want.  Note here: each FST in "nbest1.JOB.gz" is a linear FST,
      # it has no alternatives (the N-best format works by having multiple keys
      # for each utterance).  When we do "lattice-1best" we are selecting the best
      # path through the LM, there are no alternatives to consider within the
      # original lattice.
      $cmd JOB=1:$nj $dir/log/remove_old.JOB.log \
        lattice-scale --acoustic-scale=-1 --lm-scale=-1 "ark:gunzip -c $dir/nbest1.JOB.gz|" ark:- \| \
        lattice-compose ark:- "fstproject --project_output=true $oldlm |" ark:- \| \
        lattice-1best ark:- ark:- \| \
        lattice-scale --acoustic-scale=-1 --lm-scale=-1 ark:- "ark:|gzip -c >$dir/nbest2.JOB.gz" \
        || exit 1;
    fi
  fi
else
  if [ $stage -le 2 ]; then
    echo "$0: removing old LM scores."
    $cmd JOB=1:$nj $dir/log/remove_old.JOB.log \
      lattice-lmrescore-const-arpa --lm-scale=-1.0 \
      "ark:gunzip -c $dir/nbest1.JOB.gz|" $oldlm \
      "ark:|gzip -c >$dir/nbest2.JOB.gz"  || exit 1;
  fi
fi

if [ $stage -le 3 ]; then
# Decompose the n-best lists into 4 archives.
  echo "$0: creating separate-archive form of N-best lists."
  $cmd JOB=1:$nj $dir/log/make_new_archives.JOB.log \
    mkdir -p $adir.JOB '&&' \
    nbest-to-linear "ark:gunzip -c $dir/nbest2.JOB.gz|" \
    "ark,t:$adir.JOB/ali" "ark,t:$adir.JOB/words" \
    "ark,t:$adir.JOB/lmwt.nolm" "ark,t:$adir.JOB/acwt" || exit 1;
fi

if [ $stage -le 4 ]; then
  echo "$0: doing the same with old LM scores."
# Create an archive with the LM scores before we
# removed the LM probs (will help us do interpolation).
$cmd JOB=1:$nj $dir/log/make_old_archives.JOB.log \
  nbest-to-linear "ark:gunzip -c $dir/nbest1.JOB.gz|" "ark:/dev/null" \
  "ark:/dev/null" "ark,t:$adir.JOB/lmwt.withlm" "ark:/dev/null" || exit 1;
fi

if $test; then # This branch is a sanity check that at the acwt where we generated
  # the N-best list, we get the same WER.
  echo "$0 [testing branch]: generating lattices without changing scores."
  $cmd JOB=1:$nj $dir/log/test.JOB.log \
    linear-to-nbest "ark:$adir.JOB/ali" "ark:$adir.JOB/words" "ark:$adir.JOB/lmwt.withlm" \
     "ark:$adir.JOB/acwt" ark:- \| \
    nbest-to-lattice ark:- "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1;
  exit 0;
fi

if [ $stage -le 5 ]; then
  echo "$0: Creating archives with text-form of words, and LM scores without graph scores."
    # Do some small tasks; for these we don't use the queue, it will only slow us down.
  for n in $(seq $nj); do
    utils/int2sym.pl -f 2- $oldlang/words.txt < $adir.$n/words > $adir.$n/words_text || exit 1;
    mkdir -p $adir.$n/temp
    paste $adir.$n/lmwt.nolm $adir.$n/lmwt.withlm | awk '{print $1, ($4-$2);}' > \
      $adir.$n/lmwt.lmonly || exit 1;
  done
fi

if [ $stage -le 6 ]; then
  echo "$0: invoking steps/pytorchnn/compute_sentence_scores.py which computes sentence scores with a PyTorch trained neural LM."
  $cmd JOB=1:$nj $dir/log/compute_sentence_scores_pytorchnn.JOB.log \
    PYTHONPATH=steps/pytorchnn python steps/pytorchnn/compute_sentence_scores_bayes_jianwei.py \
        --nbest-list $adir.JOB/words_text \
        --outfile $adir.JOB/lmwt.nn \
        --vocabulary $vocabulary \
        --model-path $nn_model \
        --model $model_type \
        --emsize $embedding_dim \
        --nhid $hidden_dim \
        --nlayers $nlayers \
        --nhead $nhead \
        --uncertainty $uncertainty \
        --L_bayes_pos $L_bayes_pos \
        --T_bayes_pos $T_bayes_pos \
        --L_v_pos $L_v_pos \
        --T_v_pos $T_v_pos \
        --L_gauss_pos $L_gauss_pos \
        --T_gauss_pos $T_gauss_pos \
        --interpolation_flag $interpolation_flag \
        --inter_alpha $inter_alpha
fi

if [ $stage -le 7 ]; then
  echo "$0: reconstructing total LM+graph scores including interpolation of neural LM and old LM scores."
  for n in $(seq $nj); do
    paste $adir.$n/lmwt.nolm $adir.$n/lmwt.lmonly $adir.$n/lmwt.nn | awk -v nnweight=$nnweight \
      '{ key=$1; graphscore=$2; lmscore=$4; nnscore=$6;
     score = graphscore+(nnweight*nnscore)+((1-nnweight)*lmscore);
     print $1,score; } ' > $adir.$n/lmwt.interp.$nnweight || exit 1;
  done
fi

if [ $stage -le 8 ]; then
  echo "$0: reconstructing archives back into lattices."
  $cmd JOB=1:$nj $dir/log/reconstruct_lattice.JOB.log \
    linear-to-nbest "ark:$adir.JOB/ali" "ark:$adir.JOB/words" \
    "ark:$adir.JOB/lmwt.interp.$nnweight" "ark:$adir.JOB/acwt" ark:- \| \
    nbest-to-lattice ark:- "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1;
fi

if ! $skip_scoring ; then
  echo "scoring..."
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score_asclite.sh --cmd "$cmd" $data $oldlang $dir ||
    { echo "$0: Scoring failed. (ignore by '--skip-scoring true')"; exit 1; }
fi

exit 0;
