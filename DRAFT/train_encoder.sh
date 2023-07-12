python -m torch.distributed.launch --nproc_per_node=3 train_dense_encoder.py \
train_datasets=[nq_train] \
dev_datasets=[nq_dev] \
train=biencoder_local \
output_dir=/output/128tokens_simcse_ckpt

# pretrained_model_cfg= princeton-nlp/sup-simcse-bert-base-uncased
# encoder_model_type = hf_simcse