#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""
import glob
import logging
import pickle
import time
from xmlrpc.client import boolean
from typing import List, Tuple, Dict, Iterator
import random
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T
from torch import nn
import torch.nn.functional as F
from dpr.utils.data_utils import RepTokenSelector
from dpr.data.retriever_data import KiltCsvCtxSrc, TableChunk
from dpr.indexer.faiss_indexers import (
    DenseIndexer,
)
from dpr.models import init_biencoder_components
from dpr.models.biencoder import (
    BiEncoder,
    _select_span_with_token,
)
from dpr.options import setup_logger, setup_cfg_gpu, set_cfg_params_from_state
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint


logger = logging.getLogger()
setup_logger(logger)



def get_all_passages(ctx_sources):
    all_passages = {}
    for ctx_src in ctx_sources:
        ctx_src.load_data_to(all_passages)
        logger.info("Loaded ctx data: %d", len(all_passages))
        print(len(all_passages))

    if len(all_passages) == 0:
        raise RuntimeError("No passages data found. Please specify ctx_file param properly.")
    return all_passages

def iterate_encoded_files(vector_files: list, path_id_prefixes: List = None) -> Iterator[Tuple]:
    for i, file in enumerate(vector_files):
        logger.info("Reading file %s", file)
        id_prefix = None
        if path_id_prefixes:
            id_prefix = path_id_prefixes[i]
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                doc = list(doc)
                if id_prefix and not str(doc[0]).startswith(id_prefix):
                    doc[0] = id_prefix + str(doc[0])
                yield doc

def generate_question_vectors(
    question_encoder: torch.nn.Module,
    tensorizer: Tensorizer,
    questions: List[str],
    bsz: int,
    query_token: str = None,
    selector: RepTokenSelector = None,
) -> T:
    n = len(questions)
    query_vectors = []

    with torch.no_grad():
        for j, batch_start in enumerate(range(0, n, bsz)):
            batch_questions = questions[batch_start : batch_start + bsz]

            if query_token:

                if query_token == "[START_ENT]":
                    batch_tensors = [
                        _select_span_with_token(q, tensorizer, token_str=query_token) for q in batch_questions
                    ]
                else:
                    batch_tensors = [tensorizer.text_to_tensor(" ".join([query_token, q])) for q in batch_questions]
            elif isinstance(batch_questions[0], T):
                batch_tensors = [q for q in batch_questions]
            else:
                batch_tensors = [tensorizer.text_to_tensor(q) for q in batch_questions]

            max_vector_len = max(q_t.size(0) for q_t in batch_tensors)
            min_vector_len = min(q_t.size(0) for q_t in batch_tensors)

            if max_vector_len != min_vector_len:

                from dpr.models.reader import _pad_to_len
                batch_tensors = [_pad_to_len(q.squeeze(0), 0, max_vector_len) for q in batch_tensors]

            q_ids_batch = torch.stack(batch_tensors, dim=0).cuda()
            q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
            q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)

            if selector:
                rep_positions = selector.get_positions(q_ids_batch, tensorizer)

                _, out, _ = BiEncoder.get_representation(
                    question_encoder,
                    q_ids_batch,
                    q_seg_batch,
                    q_attn_mask,
                    representation_token_pos=rep_positions,
                )
            else:
                _, out, _ = question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

            query_vectors.extend(out.cpu().split(1, dim=0))



    query_tensor = torch.cat(query_vectors, dim=0)

    assert query_tensor.size(0) == len(questions)
    return query_tensor

class DenseRetriever(object):
    def __init__(self, question_encoder: nn.Module, batch_size: int, tensorizer: Tensorizer):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.selector = None

    def generate_question_vectors(self, questions: List[str], query_token: str = None) -> T:

        bsz = self.batch_size
        self.question_encoder.eval()
        return generate_question_vectors(
            self.question_encoder,
            self.tensorizer,
            questions,
            bsz,
            query_token=query_token,
            selector=self.selector,
        )

class LocalFaissRetriever(DenseRetriever):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        tensorizer: Tensorizer,
        index: DenseIndexer,
    ):
        super().__init__(question_encoder, batch_size, tensorizer)
        self.index = index

    def index_encoded_data(
        self,
        vector_files: List[str],
        buffer_size: int,
        path_id_prefixes: List = None,
    ):
        """
        Indexes encoded passages takes form a list of files
        :param vector_files: file names to get passages vectors from
        :param buffer_size: size of a buffer (amount of passages) to send for the indexing at once
        :return:
        """
        buffer = []
        for i, item in enumerate(iterate_encoded_files(vector_files, path_id_prefixes=path_id_prefixes)):
            buffer.append(item)
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        self.index.index_data(buffer)
        logger.info("Data indexing completed.")

    def get_top_docs(self, query_vectors: np.array, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """


        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info("index search time: %f sec.", time.time() - time0)
        
        return results

    def search_with_L2(self, query_vectors: np.array, radius: int = 1):

        time0 = time.time()
        results = self.index.search_with_radius(query_vectors, radius) # range_search
        logger.info("index search time: %f sec.", time.time() - time0)
        
        return results



# =============================================================================================================================

@hydra.main(config_path="conf", config_name="dense_retriever")
def main(cfg: DictConfig):

    path = '../../retrieved_text_output/'

    for num_loop in range(1):
        print(num_loop)
        if num_loop == 0:
    # ================simcse - cos_sim======================
            # you should choose settings for experiment with these parameters.

             # simcse-based cos-sim (min)
             include_random_neg = False
             with_neg_query = False
             heuristic = 'no' 
             heuristic_N = 0 
             standard_distance = 'min' 
             distance_metric = 'cos_sim' 
             pos_output_path = path + 'simcse/cos_sim/cos_sim_min_simcse_positive.csv' 
             neg_output_path = path + 'simcse/cos_sim/cos_sim_min_simcse_negative.csv' 
        
        batch_size = 1024
        
        cfg = setup_cfg_gpu(cfg)

        # using BERT-based dual-encoder
        # saved_state = load_states_from_checkpoint('../../donwloaded_data/downloads/checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp')
               
        # using SimCSE-based dual-encoder
        saved_state = load_states_from_checkpoint("../../output/simcse_ckpt/dpr_biencoder.32")

        set_cfg_params_from_state(saved_state.encoder_params, cfg)

        logger.info("CFG (after gpu  configuration):")
        logger.info("%s", OmegaConf.to_yaml(cfg))

        tensorizer, encoder, _ = init_biencoder_components(cfg.encoder.encoder_model_type, cfg, inference_only=True)

        logger.info("Loading saved model state ...")
        encoder.load_state(saved_state, strict=False)

        encoder_path = cfg.encoder_path
        if encoder_path:
            logger.info("Selecting encoder: %s", encoder_path)
            encoder = getattr(encoder, encoder_path)
        else:
            logger.info("Selecting standard question encoder")
            encoder = encoder.question_model

        encoder, _ = setup_for_distributed_mode(encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16)

        encoder.eval()

        model_to_load = get_model_obj(encoder)
        vector_size = model_to_load.get_out_size()
        logger.info("Encoder vector_size=%d", vector_size)

        # get questions & answers
        questions = []

        # religion queries example
        # https://www.history.com/topics/religion/
        # religion positive queries
        total_queries = ['Judaism is the world’s oldest monotheistic religion, dating back nearly 4,000 years. ',
                         'Christianity is the most widely practiced religion in the world, with more than 2 billion followers. ',
                         'Islam is the second largest religion in the world after Christianity, with about 1.8 billion Muslims worldwide.',
                        #  'Buddhism is a faith that was founded by Siddhartha Gautama (“the Buddha”) more than 2,500 years ago in India.',
                        #  'Hinduism is the world’s oldest religion, according to many scholars, with roots and customs dating back more than 4,000 years. '
                         ]
        # religion negative queries
        negative_queries = [
                        'Buddhism is a faith that was founded by Siddhartha Gautama (“the Buddha”) more than 2,500 years ago in India.',
                        'Hinduism is the world’s oldest religion, according to many scholars, with roots and customs dating back more than 4,000 years.'
        ]


        # South Korea keyword
        # south korea positive queries
        # total_queries = ['South Korea', 'K-pop', 'Kimchi', 'Seoul'] 

        # south korea negative queries
        # negative_queries = ['North Korea', 'Japan', 'China']

        # Toyota queries examples (keyword + sentence)
        # total_queries = ["Toyota Motor Corporation",
        #                 "Akio Toyoda",
        #                 "Toyota will lead the future mobility society, enriching lives around the world with the safest and most responsible ways of moving people.",
        #                 "Combining software, hardware and partnerships to create unique value that comes from the Toyota Way."]
        


        for query in total_queries:
            questions.append(query)

        index = hydra.utils.instantiate(cfg.indexers[cfg.indexer])
        logger.info("Local Index class %s ", type(index))
        index_buffer_sz = index.buffer_size
        index.init_index(vector_size)

        print('retriever LocalFaissRetriever')
        retriever = LocalFaissRetriever(encoder, batch_size, tensorizer, index)

        questions_tensor = retriever.generate_question_vectors(questions, query_token=None)

        index_path = cfg.index_path # null

        # send data for indexing
        id_prefixes = []
        ctx_sources = []

        for ctx_src in ['dpr_wiki']:
            ctx_src = hydra.utils.instantiate(cfg.ctx_sources[ctx_src])
            id_prefixes.append(ctx_src.id_prefix)
            ctx_sources.append(ctx_src)
            logger.info("ctx_sources: %s", type(ctx_src))

        logger.info("id_prefixes per dataset: %s", id_prefixes)

        # index all passages       
        # BERT-based embedding
        # ctx_files_patterns = ['../../donwloaded_data/downloads/data/retriever_results/nq/single-adv-hn/wikipedia_passages_*']

        # SimCSE-based embedding
        ctx_files_patterns = ["../../output/DenseEmbedding/SimCSE_embedding_0"]


        logger.info("ctx_files_patterns: %s", ctx_files_patterns)
        if ctx_files_patterns:
            assert len(ctx_files_patterns) == len(id_prefixes), "ctx len={} pref leb={}".format(
                len(ctx_files_patterns), len(id_prefixes)
            )
        else:
            assert (
                index_path or cfg.rpc_index_id
            ), "Either encoded_ctx_files or index_path pr rpc_index_id parameter should be set."

        input_paths = []
        path_id_prefixes = []
        for i, pattern in enumerate(ctx_files_patterns):
            pattern_files = glob.glob(pattern)
            pattern_id_prefix = id_prefixes[i]
            input_paths.extend(pattern_files)
            path_id_prefixes.extend([pattern_id_prefix] * len(pattern_files))
            
        logger.info("Embeddings files id prefixes: %s", path_id_prefixes)
        logger.info("Reading all passages data from files: %s", input_paths)
        print('\n')


        # enc -> get collection db for text
        all_passages = get_all_passages(ctx_sources)
        
        retriever.index_encoded_data(input_paths, index_buffer_sz, path_id_prefixes=path_id_prefixes)
        if index_path:
            retriever.index.serialize(index_path)


        # heuristic method
        if heuristic == 'yes':

            print(f'get top {heuristic_N} passages start!!!!!!')
            print('question tensor: ',questions_tensor.numpy())
            top_results_and_scores = retriever.get_top_docs(query_vectors = questions_tensor.numpy(), top_docs = heuristic_N)
            print(f'get top {heuristic_N} passages done!!!!!!')
        
            retrieved_texts = []
            for i, (content, score) in enumerate(top_results_and_scores):
                tmp_text = []
                for k in content:
                    text = all_passages[k]
                    tmp_text.append(text[0])

                retrieved_texts.append(tmp_text)

            retrieved_texts = sum(retrieved_texts,[])

        
        else:
            
            # cosine sim 
            if distance_metric == 'cos_sim':
                cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
            
                dist_list = []
                for i, text_a in enumerate(questions_tensor):
                    for j, text_b in enumerate(questions_tensor):
                        if i != j and i < j:
                            dist = cos(torch.tensor(text_a),torch.tensor(text_b))
                            dist_list.append(dist)

                if standard_distance == 'avg':
                    standard_dist = np.average(dist_list)
                elif standard_distance == 'min':
                    standard_dist = np.min(dist_list)
                elif standard_distance == 'median':
                    standard_dist = np.median(dist_list)
                elif standard_distance == 'max':
                    standard_dist = np.max(dist_list)

                logger.info("standard_dist: %s", str(standard_dist))

                top_results_and_scores = retriever.get_top_docs(query_vectors = questions_tensor.numpy(), top_docs = 10000)

                retrieved_texts = []
                for i, (content, score) in enumerate(top_results_and_scores):
                    print(f'{i}번째 retrieved texts')

                    query = questions_tensor[i]

                    tmp_text = []
                    for k in content:
                        text = all_passages[k]    
                        text_emb = retriever.generate_question_vectors(text, query_token=None) 
                        cos_sim = cos(torch.tensor(query),torch.tensor(text_emb[0]))
                        if  cos_sim > standard_dist:
                            tmp_text.append(text[0])
                        else:
                            continue

                    retrieved_texts.append(tmp_text)

                retrieved_texts = sum(retrieved_texts,[])
                print('positive samples retrieving clear!')



            # dot product
            elif distance_metric == 'dot_product':

                dist_list = []
                for i, text_a in enumerate(questions_tensor):
                    for j, text_b in enumerate(questions_tensor):
                        if i != j and i < j:
                            inner_product = torch.dot(torch.tensor(text_a), torch.tensor(text_b))
                            dist_list.append(inner_product)

                if standard_distance == 'avg':
                    standard_dist = np.average(dist_list)
                elif standard_distance == 'min':
                    standard_dist = np.min(dist_list)
                elif standard_distance == 'median':
                    standard_dist = np.median(dist_list)
                elif standard_distance == 'max':
                    standard_dist = np.max(dist_list)

                logger.info("standard_dist: %s", str(standard_dist))

                l2_based_results = retriever.search_with_L2(query_vectors = questions_tensor.numpy(),radius=standard_dist)

                retrieved_texts = []
                for ids in l2_based_results:
                    text = all_passages[ids][0]
                    retrieved_texts.append(text)
                



        # negative query input 
        if with_neg_query:
            neq_q = []
            for neg in negative_queries:
                neq_q.append(neg)
            
            neq_questions_tensor = retriever.generate_question_vectors(neq_q, query_token=None)
            
            neg_dist_list = []
            for i, neg_text_a in enumerate(neq_questions_tensor):
                for j, neg_text_b in enumerate(neq_questions_tensor):
                    if i != j and i < j:
                        neg_dist = cos(torch.tensor(neg_text_a),torch.tensor(neg_text_b))
                        neg_dist_list.append(neg_dist)

            if standard_distance == 'avg':
                neg_standard_dist = np.average(neg_dist_list)
            elif standard_distance == 'min':
                neg_standard_dist = np.min(neg_dist_list)
            elif standard_distance == 'median':
                neg_standard_dist = np.median(neg_dist_list)
            elif standard_distance == 'max':
                neg_standard_dist = np.max(neg_dist_list)

            logger.info("standard_dist: %s", str(neg_standard_dist))

            neg_top_results_and_scores = retriever.get_top_docs(query_vectors = neq_questions_tensor.numpy(), top_docs = 10000)

            neg_retrieved_texts = []
            for i, (content, score) in enumerate(neg_top_results_and_scores):
                print(f'{i}th neg query retrieved texts')

                query = neq_questions_tensor[i]

                tmp_text = []
                for k in content:
                    text = all_passages[k]   
                    text_emb = retriever.generate_question_vectors(text, query_token=None) 
                    cos_sim = cos(torch.tensor(query),torch.tensor(text_emb[0]))
                    if  cos_sim > standard_dist:
                        tmp_text.append(text[0])
                    else:
                        continue

                neg_retrieved_texts.append(tmp_text)

            neg_retrieved_texts = sum(neg_retrieved_texts,[])
            print('negative samples query based retrieving clear!')

            if include_random_neg:
                print('negative samples randomly retrievs start')
                neg_num = len(retrieved_texts) // 2 # 50% random sample
                print(neg_num, ' number of samples will be sampled')
                keys = random.sample(list(all_passages), neg_num)           
                negative_texts = [all_passages[k][0] for k in keys]

                cnt = 0
                for j,text in enumerate(negative_texts):
                    if text in retrieved_texts or 'wiki:' in negative_texts[j]:
                        negative_texts.pop(j)
                        cnt += 1
                    elif text in neg_retrieved_texts or 'wiki:' in negative_texts[j]:
                        negative_texts.pop(j)
                        cnt += 1
                if cnt >= 1:
                    new_keys = random.sample(list(all_passages.values()), cnt)
                    negative_texts.extend(new_keys)
                print(f'duplicated {cnt}')

                # random 50% sampling
                neg_dict_random = {'text':negative_texts}
                neg_df_random = pd.DataFrame(neg_dict_random)

                # neg query 50% sampling
                neg_dict = {'text':neg_retrieved_texts}
                neg_df = pd.DataFrame(neg_dict)
                neg_df = neg_df.sample(n=neg_df.shape[0]//2, random_state=1234)


                neg_df = pd.concat([neg_df,neg_df_random],axis=0)
                print(f'negative number {neg_df.shape[0]}')
                neg_df.to_csv(neg_output_path)

                pos_dict = {'text':retrieved_texts}
                pos_df = pd.DataFrame(pos_dict)
                print(f'positive number {pos_df.shape[0]}')
                pos_df.to_csv(pos_output_path)

            else:            
                pos_dict = {'text':retrieved_texts}
                neg_dict = {'text':neg_retrieved_texts}
                pos_df = pd.DataFrame(pos_dict)
                neg_df = pd.DataFrame(neg_dict)

                print(f'positive number {pos_df.shape[0]}')
                pos_df.to_csv(pos_output_path)

                print(f'negative number {neg_df.shape[0]}')
                neg_df.to_csv(neg_output_path)

        # only random sample
        else:   
            print('negative samples retrievs start')
            neg_num = len(retrieved_texts)
            print(neg_num, ' number of samples will be sampled')
            keys = random.sample(list(all_passages), neg_num)           
            negative_texts = [all_passages[k][0] for k in keys]

            cnt = 0
            for j,text in enumerate(negative_texts):
                if text in retrieved_texts or 'wiki:' in negative_texts[j]:
                    negative_texts.pop(j)
                    cnt += 1
            if cnt >= 1:
                new_keys = random.sample(list(all_passages.values()), cnt+10)
                negative_texts.extend(new_keys)
            print(f'duplicated {cnt}')


            pos_dict = {'text':retrieved_texts}
            pos_df = pd.DataFrame(pos_dict)
            print(f'positive number {pos_df.shape[0]}')
            pos_df.to_csv(pos_output_path)

            neg_dict = {'text':negative_texts}
            neg_df = pd.DataFrame(neg_dict)
            print(f'negative number {neg_df.shape[0]}')
            neg_df.to_csv(neg_output_path)


if __name__ == "__main__":
    main()