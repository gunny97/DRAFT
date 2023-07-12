python generate_dense_embeddings.py \
	--model_file=output/simcse_ckpt/dpr_biencoder.32 \
	--ctx_src=dpr_wiki \
	--shard_id=0 num_shards=1 \
	--out_file=output/DenseEmbedding/SimCSE_embedding