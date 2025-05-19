HF_MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B # path to your download HF model
CKPT_PATH=./ckpts/adapt_think_ds1.5b_deepscaler_btz128_n16_nr0.5-sl16384-fl4096-nb0.05-lr2e-6/global_step_300/actor
SAVE_PATH=./ckpts/adapt_think_ds1.5b_deepscaler_btz128_n16_nr0.5-sl16384-fl4096-nb0.05-lr2e-6/global_step_300/HF
python src/model_merger.py --backend fsdp --hf_model_path $HF_MODEL_PATH --local_dir $CKPT_PATH --target_dir $SAVE_PATH

TOKENIZER_FILES=("tokenizer_config.json" "tokenizer.json")
for FILE in "${TOKENIZER_FILES[@]}"; do
    cp $HF_MODEL_PATH/$FILE $SAVE_PATH
done
