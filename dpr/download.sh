echo "Donwload new model DPR ckpt"
python data/download_data.py --resource checkpoint.retriever.single-adv-hn.nq.bert-base-encoder --output_dir /home/local/anaconda3/envs/paper/DPR/donwloaded_data


echo "Donwload wikipedia embeddings"
python data/download_data.py --resource data.retriever_results.nq.single-adv-hn.wikipedia_passages --output_dir /home/local/anaconda3/envs/paper/DPR/donwloaded_data
# psgs_w100.tsv 아님

echo "Download training dataset"
python data/download_data.py --resource data.retriever.nq-adv-hn-train --output_dir /home/local/anaconda3/envs/paper/DPR/donwloaded_data

echo "Download test dataset"
python data/download_data.py --resource data.retriever_results.nq.single-adv-hn.test --output_dir /home/local/anaconda3/envs/paper/DPR/donwloaded_data



