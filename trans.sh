SRC_LANG='kr'
TRG_LANG='en'
BPE=1

RNN='mylstm'

DIM_WEMB=300
DIM_ENC=500
DIM_ATT=500
DIM_DEC=500

#TRAIN_GPU=3
BEAM_WIDTH=1

DICT1='vocab.'$SRC_LANG'.pkl'
DICT2='vocab.'$TRG_LANG'.pkl'

if [ $SRC_LANG == 'kr' -o $TRG_LANG == 'kr' ]
then
    DATA_DIR='../data_process'
    VALID_FILE1='valid.'$SRC_LANG'.tok'
    VALID_FILE2='valid.'$TRG_LANG'.tok'
    SUB_DIR='subword'

fi

VALID_FILE2=$DATA_DIR'/'$VALID_FILE2
if [ $BPE == 1 ]
then
    DATA_DIR=$DATA_DIR'/'$SUB_DIR
    VALID_FILE1=$VALID_FILE1'.sub'
fi
TRANS_FILE=$VALID_FILE1'.trans'
VALID_FILE1=$DATA_DIR'/'$VALID_FILE1

DICT1=$DATA_DIR'/'$DICT1
DICT2=$DATA_DIR'/'$DICT2

SAVE_DIR='./results'
#mkdir $SAVE_DIR'/trans'

MODEL_FILE=$SRC_LANG'2'$TRG_LANG'.'$RNN'.'$DIM_WEMB'.'$DIM_ENC'.'$DIM_ATT'.'$DIM_DEC
USE_BEST=1

#VALID_FILE1='/home/nmt19/data_process/subword/valid.kr.tok.sub'
VALID_FILE1='/home/nmt19/data_process/subword/valid.kr.tok.sub'
VALID_FLIE2='/home/nmt19/data_process/valid.en.tok'
DICT1='/home/nmt19/data_process/subword/vocab.kr.pkl'
DICT2='/home/nmt19/data_process/subword/vocab.en.pkl'
#TRANS_FILE=$VALID_FILE1'.trans'

TRANS_FILE=$SAVE_DIR'/trans/'$MODEL_FILE'-'$TRANS_FILE
CUDA_VISIBLE_DEVICES=$1 python3 nmt_run.py --train=0 --trans=1 --rnn_name=$RNN \
        --save_dir=$SAVE_DIR --model_file=$MODEL_FILE --trans_file=$TRANS_FILE \
        --valid_src_file=$VALID_FILE1 --valid_trg_file=$VALID_FILE2 \
        --src_dict=$DICT1 --trg_dict=$DICT2 --use_best=$USE_BEST \
        --beam_width=$BEAM_WIDTH

#perl multi-bleu.perl $VALID_FILE2 <  $TRANS_FILE
