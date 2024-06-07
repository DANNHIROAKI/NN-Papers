#!/bin/bash

. cmd.sh
. path.sh

# This setup was modified from egs/swbd/s5b, with the following changes:
# 1. added more training data for early stages
# 2. removed SAT system (and later stages) on the 100k utterance training data
# 3. reduced number of LM rescoring, only sw1_tg and sw1_fsh_fg remain
# 4. mapped swbd transcription to fisher style, instead of the other way around 

set -e # exit on error
has_fisher=true
local/swbd1_data_download.sh /data/sls/static/switchboard
# prepare SWBD dictionary first since we want to find acronyms according to pronunciations
# before mapping lexicon and transcripts
local/swbd1_prepare_dict.sh

# Prepare Switchboard data. This command can also take a second optional argument
# which specifies the directory to Switchboard documentations. Specifically, if
# this argument is given, the script will look for the conv.tab file and correct
# speaker IDs to the actual speaker personal identification numbers released in
# the documentations. The documentations can be found here: 
# https://catalog.ldc.upenn.edu/docs/LDC97S62/
# Note: if you are using this link, make sure you rename conv_tab.csv to conv.tab
# after downloading.
# Usage: local/swbd1_data_prep.sh /path/to/SWBD [/path/to/SWBD_docs]
local/swbd1_data_prep.sh /data/sls/static/switchboard
# local/swbd1_data_prep.sh /home/dpovey/data/LDC97S62
# local/swbd1_data_prep.sh /data/corpora0/LDC97S62
# local/swbd1_data_prep.sh /mnt/matylda2/data/SWITCHBOARD_1R2
# local/swbd1_data_prep.sh /exports/work/inf_hcrc_cstr_general/corpora/switchboard/switchboard1

utils/prepare_lang.sh data/local/dict \
  "<unk>"  data/local/lang_nopp data/lang_nopp

# Now train the language models. We are using SRILM and interpolating with an
# LM trained on the Fisher transcripts (part 2 disk is currently missing; so 
# only part 1 transcripts ~700hr are used)

# If you have the Fisher data, you can set this "fisher_dir" variable.
fisher_dirs="/data/sls/static/LDC2004T19/fe_03_p1_tran/ /data/sls/static/LDC2005T19/fe_03_p2_tran/"
#fisher_dirs="/home/dpovey/data/LDC2004T19/fe_03_p1_tran/"
#fisher_dirs="/data/corpora0/LDC2004T19/fe_03_p1_tran/"
# edinburgh:
# fisher_dirs="/exports/work/inf_hcrc_cstr_general/corpora/fisher/transcripts"
# brno:
# fisher_dirs="/mnt/matylda2/data/FISHER/fe_03_p1_tran" # BUT
local/swbd1_train_lms.sh data/local/train/text \
  data/local/dict/lexicon.txt data/local/lm $fisher_dirs

# Compiles G for swbd trigram LM
LM=data/local/lm/sw1.o3g.kn.gz
srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"
utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
  data/lang_nopp $LM data/local/dict/lexicon.txt data/lang_nopp_sw1_tg

# Compiles const G for swbd+fisher 4gram LM, if it exists.
LM=data/local/lm/sw1_fsh.o4g.kn.gz
[ -f $LM ] || has_fisher=false
if $has_fisher; then
  utils/build_const_arpa_lm.sh $LM data/lang_nopp data/lang_nopp_sw1_fsh_fg
fi

# Data preparation and formatting for eval2000 (note: the "text" file
# is not very much preprocessed; for actual WER reporting we'll use
# sclite.

# local/eval2000_data_prep.sh /data/corpora0/LDC2002S09/hub5e_00 /data/corpora0/LDC2002T43
# local/eval2000_data_prep.sh /mnt/matylda2/data/HUB5_2000/ /mnt/matylda2/data/HUB5_2000/2000_hub5_eng_eval_tr
# local/eval2000_data_prep.sh /exports/work/inf_hcrc_cstr_general/corpora/switchboard/hub5/2000 /exports/work/inf_hcrc_cstr_general/corpora/switchboard/hub5/2000/transcr
# local/eval2000_data_prep.sh /home/dpovey/data/LDC2002S09/hub5e_00 /home/dpovey/data/LDC2002T43
local/eval2000_data_prep.sh /data/sls/static/LDC2002S09/hub5e_00 /data/sls/static/LDC2002T43/2000_hub5_eng_eval_tr

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc
for x in train eval2000; do
  steps/make_mfcc.sh --nj 50 --cmd "$train_cmd" \
    data/$x exp/make_mfcc/$x $mfccdir
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir 
  utils/fix_data_dir.sh data/$x
done

# Use the first 4k sentences as dev set.  Note: when we trained the LM, we used
# the 1st 10k sentences as dev set, so the 1st 4k won't have been used in the
# LM training data.   However, they will be in the lexicon, plus speakers
# may overlap, so it's still not quite equivalent to a test set.
utils/subset_data_dir.sh --first data/train 4000 data/train_dev # 5hr 6min
n=$[`cat data/train/segments | wc -l` - 4000]
utils/subset_data_dir.sh --last data/train $n data/train_nodev

# Now-- there are 260k utterances (313hr 23min), and we want to start the 
# monophone training on relatively short utterances (easier to align), but not 
# only the shortest ones (mostly uh-huh).  So take the 100k shortest ones;
# remove most of the repeated utterances (these are the uh-huh type ones), and 
# then take 10k random utterances from those (about 4hr 40mins)
utils/subset_data_dir.sh --shortest data/train_nodev 100000 data/train_100kshort
utils/subset_data_dir.sh data/train_100kshort 30000 data/train_30kshort

# Take the first 100k utterances (just under half the data); we'll use
# this for later stages of training.
utils/subset_data_dir.sh --first data/train_nodev 100000 data/train_100k
local/remove_dup_utts.sh 200 data/train_100k data/train_100k_nodup  # 110hr

# Finally, the full training set:
local/remove_dup_utts.sh 300 data/train_nodev data/train_nodup  # 286hr
## Starting basic training on MFCC features
steps/train_mono.sh --nj 30 --cmd "$train_cmd" \
  data/train_30kshort data/lang_nopp exp/mono 

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_100k_nodup data/lang_nopp exp/mono exp/mono_ali 

steps/train_deltas.sh --cmd "$train_cmd" \
  3200 30000 data/train_100k_nodup data/lang_nopp exp/mono_ali exp/tri1 

(
  graph_dir=exp/tri1/graph_nopp_sw1_tg
  $train_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_nopp_sw1_tg exp/tri1 $graph_dir
  steps/decode_si.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
    $graph_dir data/eval2000 exp/tri1/decode_eval2000_nopp_sw1_tg
) &

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_100k_nodup data/lang_nopp exp/tri1 exp/tri1_ali 

steps/train_deltas.sh --cmd "$train_cmd" \
  4000 70000 data/train_100k_nodup data/lang_nopp exp/tri1_ali exp/tri2

(
  # The previous mkgraph might be writing to this file.  If the previous mkgraph
  # is not running, you can remove this loop and this mkgraph will create it.
  while [ ! -s data/lang_nopp_sw1_tg/tmp/CLG_3_1.fst ]; do sleep 60; done
  sleep 20; # in case still writing.
  graph_dir=exp/tri2/graph_nopp_sw1_tg
  $train_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_nopp_sw1_tg exp/tri2 $graph_dir
  steps/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
    $graph_dir data/eval2000 exp/tri2/decode_eval2000_nopp_sw1_tg
) &

# The 100k_nodup data is used in neural net training.
steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_100k_nodup data/lang_nopp exp/tri2 exp/tri2_ali_100k_nodup

# From now, we start using all of the data (except some duplicates of common
# utterances, which don't really contribute much).
steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_nodup data/lang_nopp exp/tri2 exp/tri2_ali_nodup 

# Do another iteration of LDA+MLLT training, on all the data.
steps/train_lda_mllt.sh --cmd "$train_cmd" \
  6000 140000 data/train_nodup data/lang_nopp exp/tri2_ali_nodup exp/tri3 

(
  graph_dir=exp/tri3/graph_nopp_sw1_tg
  $train_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_nopp_sw1_tg exp/tri3 $graph_dir
  steps/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
    $graph_dir data/eval2000 exp/tri3/decode_eval2000_nopp_sw1_tg
) &
# Now we compute the pronunciation probabilities from training data, and
# re-create the lang directory.
steps/get_prons.sh --cmd "$train_cmd" data/train_nodup data/lang_nopp exp/tri3
utils/dict_dir_add_pronprobs.sh --max-normalize true \
  data/local/dict exp/tri3/pron_counts_nowb.txt data/local/dict_pp

utils/prepare_lang.sh data/local/dict_pp "<unk>" data/local/lang data/lang
LM=data/local/lm/sw1.o3g.kn.gz
srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"
utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
  data/lang $LM data/local/dict_pp/lexicon.txt data/lang_sw1_tg
LM=data/local/lm/sw1_fsh.o4g.kn.gz
if $has_fisher; then
  utils/build_const_arpa_lm.sh $LM data/lang data/lang_sw1_fsh_fg
fi

(
  graph_dir=exp/tri3/graph_sw1_tg
  $train_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_sw1_tg exp/tri3 $graph_dir
  steps/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
    $graph_dir data/eval2000 exp/tri3/decode_eval2000_sw1_tg
) &

# Train tri4, which is LDA+MLLT+SAT, on all the (nodup) data.
steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_nodup data/lang exp/tri3 exp/tri3_ali_nodup 


steps/train_sat.sh  --cmd "$train_cmd" \
  11500 200000 data/train_nodup data/lang exp/tri3_ali_nodup exp/tri4

(
  graph_dir=exp/tri4/graph_sw1_tg
  $train_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_sw1_tg exp/tri4 $graph_dir
  steps/decode_fmllr.sh --nj 30 --cmd "$decode_cmd" \
    --config conf/decode.config \
    $graph_dir data/eval2000 exp/tri4/decode_eval2000_sw1_tg
) &
wait

if $has_fisher; then
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
    data/lang_sw1_{tg,fsh_fg} data/eval2000 \
    exp/tri4/decode_eval2000_sw1_{tg,fsh_fg}
fi

# MMI training starting from the LDA+MLLT+SAT systems on all the (nodup) data. 
steps/align_fmllr.sh --nj 50 --cmd "$train_cmd" \
  data/train_nodup data/lang exp/tri4 exp/tri4_ali_nodup


