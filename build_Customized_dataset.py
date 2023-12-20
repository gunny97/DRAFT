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
import json
import logging
import pickle
import time
from xmlrpc.client import boolean
import zlib
from typing import List, Tuple, Dict, Iterator
import random
import csv
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T
from torch import nn
import torch.nn.functional as F
from dpr.utils.data_utils import RepTokenSelector
from dpr.data.qa_validation import calculate_matches, calculate_chunked_matches, calculate_matches_from_meta
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
import os
from datasets import load_dataset

import argparse

parser = argparse.ArgumentParser(description='build positive/negative dataset paper')

logger = logging.getLogger()
setup_logger(logger)

def seed_everything(seed: int = 1234567):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

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
                # TODO: tmp workaround for EL, remove or revise
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

            # TODO: this only works for Wav2vec pipeline but will crash the regular text pipeline
            # max_vector_len = max(q_t.size(1) for q_t in batch_tensors)
            # min_vector_len = min(q_t.size(1) for q_t in batch_tensors)
            max_vector_len = max(q_t.size(0) for q_t in batch_tensors)
            min_vector_len = min(q_t.size(0) for q_t in batch_tensors)

            if max_vector_len != min_vector_len:
                # TODO: _pad_to_len move to utils
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

            # if len(query_vectors) % 100 == 0:
                # logger.info("Encoded queries %d", len(query_vectors))

    query_tensor = torch.cat(query_vectors, dim=0)
    # logger.info("Total encoded queries tensor %s", query_tensor.size())
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
        print('start search knn !!!!!!!')
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info("index search time: %f sec.", time.time() - time0)
        # self.index = None
        return results

    def search_with_L2(self, query_vectors: np.array, radius: int = 1):

        time0 = time.time()
        print('start range_search !!!!!!!')
        results = self.index.search_with_radius(query_vectors, radius) # range_search

        logger.info("index search time: %f sec.", time.time() - time0)
        # self.index = None
        return results

# =============================================================================================================================

@hydra.main(config_path="conf", config_name="dense_retriever")
def main(cfg: DictConfig):
    seed_everything()

    path = 'DBPEDIA_AUG/'
    standard_distance = 'avg' # cfg.standard_distance / choose among 'min', 'median', 'max'
    distance_metric = 'cos_sim' # cfg.distance_metric / or choose 'dot_product'

    downstream_task_name = ['agnews','dbpedia','customized_dataset','factsnet', 'trec']
    not_need_to_check = True

    # multi
    random_seed = 1234 # 5678, 1004, 9999, 7777 
    query_number = 8
    NN_subspace = 50

    # others
    # query_number = 5
    # NN_subspace = 10000

    if not_need_to_check:
        batch_size = 4096
        
        cfg = setup_cfg_gpu(cfg)

        print('model loading --> this is bert-based')
        
        saved_state = load_states_from_checkpoint("output/simcse_ckpt/dpr_biencoder.32")

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

        index_path = cfg.index_path # null

        # send data for indexing
        id_prefixes = []
        ctx_sources = []
        # for ctx_src in cfg.ctx_datatsets:
        for ctx_src in ['dpr_wiki']:
            ctx_src = hydra.utils.instantiate(cfg.ctx_sources[ctx_src])
            id_prefixes.append(ctx_src.id_prefix)
            ctx_sources.append(ctx_src)
            logger.info("ctx_sources: %s", type(ctx_src))

        logger.info("id_prefixes per dataset: %s", id_prefixes)

        # simcse-based embedding
        ctx_files_patterns = ["output/DenseEmbedding/SimCSE_embedding_0"]


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
        print('index start!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('\n')

        # enc -> get collection db for text
        all_passages = get_all_passages(ctx_sources)    

        index = hydra.utils.instantiate(cfg.indexers[cfg.indexer])
        logger.info("Local Index class %s ", type(index))
        index_buffer_sz = index.buffer_size
        index.init_index(vector_size)

        print('retriever LocalFaissRetriever')
        retriever = LocalFaissRetriever(encoder, batch_size, tensorizer, index)
        retriever.index_encoded_data(input_paths, index_buffer_sz, path_id_prefixes=path_id_prefixes)

        if index_path:
            retriever.index.serialize(index_path)

    def constructing_dataset(total_queries, dataset_name, NN_subspace, random_seed, query_number):

        # for agnews and dbpedia dataset
        if dataset_name == 'agnews' or dataset_name == 'dbpedia'  or dataset_name == 'trec':
                            
            for LABEL_ag in np.unique(total_queries['label']):
                # get questions & answers
                questions = list(total_queries[total_queries['label']==LABEL_ag]['text'])
                logger.info("questions len %d", len(questions))

                questions_tensor = retriever.generate_question_vectors(questions, query_token=None)

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

                    top_results_and_scores = retriever.get_top_docs(query_vectors = questions_tensor.numpy(), top_docs = NN_subspace)

                    retrieved_texts = []
                    for i, (content, score) in enumerate(top_results_and_scores):
                        print(f'{i}th retrieved texts')
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
                
                retrieved_texts = sum(retrieved_texts,[])
                print('positive samples retrieving clear!')

                pos_dict = {'text':retrieved_texts}
                pos_df = pd.DataFrame(pos_dict)
                print(f'positive number {pos_df.shape[0]}')

                if dataset_name == 'dbpedia':
                    label_name = str(LABEL_ag) +  "_th_label"
                    file_path =  path + "dbpedia/"
                    os.makedirs(file_path, exist_ok=True)
                    pos_df.to_csv(file_path + f"{random_seed}_{query_number}_{label_name}.csv")

                elif dataset_name == 'agnews':
                    if LABEL_ag == 0:
                        label_name = 'world'
                    elif LABEL_ag == 1:
                        label_name = 'sports'
                    elif LABEL_ag == 2:
                        label_name = 'business'
                    elif LABEL_ag == 3:
                        label_name = 'sci_tech'

                    file_path = path + "agnews/"
                    os.makedirs(file_path, exist_ok=True)
                    pos_df.to_csv(file_path + f"{random_seed}_{query_number}_{label_name}.csv")
                

                
                elif dataset_name == 'trec':
                    label_name = f"{LABEL_ag}th_label"
                    file_path = path+ "trec/"
                    os.makedirs(file_path, exist_ok=True)
                    pos_df.to_csv(file_path + f"{random_seed}_{query_number}_{label_name}.csv")

        # customized_dataset, factsnet
        elif dataset_name == 'customized_dataset' or dataset_name == 'factsnet':
            questions = total_queries
            questions_tensor = retriever.generate_question_vectors(questions, query_token=None)

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

                top_results_and_scores = retriever.get_top_docs(query_vectors = questions_tensor.numpy(), top_docs = NN_subspace)

                retrieved_texts = []
                for i, (content, score) in enumerate(top_results_and_scores):
                    print(f'{i}th retrieved texts')

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

            return retrieved_texts

    def construct_queries(data, num, seed):
        label_queries = []
        for label in np.unique(data['label']):
            label_query = []
            cond = data['label'] == label
            query = data[cond].sample(n=num, random_state=seed)
            label_query.append(query)
            label_queries.append(label_query)
        label_queries = np.array(label_queries)
        label_queries = label_queries.squeeze(1)
        return label_queries

    for TASK_NAME in downstream_task_name:

        if TASK_NAME == 'agnews':

            ag_news = load_dataset("ag_news")
            ag_news_train_df = pd.DataFrame(ag_news['train'])

            query_result = construct_queries(ag_news_train_df, query_number, random_seed)
            query_unique_length = 4 
            total_queries = pd.DataFrame(query_result.reshape(query_unique_length*query_number,2), columns=['text','label'])
            constructing_dataset(total_queries, 'agnews', NN_subspace)



        elif TASK_NAME == 'trec':
            trec = load_dataset("trec")
            trec_train_df = pd.DataFrame(trec['train'])
            trec_train_df.drop(['label-fine'],axis=1,inplace=True)
            trec_train_df.columns = ['label','text']

            query_result = construct_queries(trec_train_df, query_number, random_seed)
            query_unique_length = 6
            total_queries = pd.DataFrame(query_result.reshape(query_unique_length*query_number,2), columns=['text','label'])
            constructing_dataset(total_queries, 'trec', NN_subspace)

        elif TASK_NAME == 'dbpedia':
            
            dataset = load_dataset("dbpedia_14")
            dbpedia_train = pd.DataFrame(dataset['train'])
            dbpedia_train.drop(['title'],axis=1,inplace=True)
            dbpedia_train.columns = ['label','text']

            for query_number in [100,50,25,5,]:
                # query_number = 8
                for random_seed in [5678]:
                    query_result = construct_queries(dbpedia_train, query_number, random_seed)
                    query_unique_length = 14
                    total_queries = pd.DataFrame(query_result.reshape(query_unique_length*query_number,2), columns=['label','text'])
                    constructing_dataset(total_queries, 'dbpedia', 10, NN_subspace, random_seed,query_number)

        elif TASK_NAME == 'customized_dataset':
            

            # religion queries example
            # https://www.history.com/topics/religion/
            
            religion_total_queries = ['Judaism is the world’s oldest monotheistic religion, dating back nearly 4,000 years. ',
                            'Christianity is the most widely practiced religion in the world, with more than 2 billion followers. ',
                            'Islam is the second largest religion in the world after Christianity, with about 1.8 billion Muslims worldwide.',
                            #  'Buddhism is a faith that was founded by Siddhartha Gautama (“the Buddha”) more than 2,500 years ago in India.',
                            #  'Hinduism is the world’s oldest religion, according to many scholars, with roots and customs dating back more than 4,000 years. '
                            ]

            religion_negative_queries = [
                            'Buddhism is a faith that was founded by Siddhartha Gautama (“the Buddha”) more than 2,500 years ago in India.',
                            'Hinduism is the world’s oldest religion, according to many scholars, with roots and customs dating back more than 4,000 years.'
            ]

            religion_questions = []
            for query in religion_total_queries:
                religion_questions.append(query)

            n_religion_questions =[]
            for n_query in religion_negative_queries:
                n_religion_questions.append(n_query)

            retrieved_texts = constructing_dataset(religion_questions, 'customized_dataset')
            n_retrieved_texts = constructing_dataset(n_religion_questions, 'customized_dataset')
            
            for negative_sample_type in ['only_random', 'only_neg_query', 'random_plus_neg_query']:
                if negative_sample_type == 'only_random':
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
                    print(f'duplicated number {cnt}')

                    pos_dict = {'text':retrieved_texts}
                    pos_df = pd.DataFrame(pos_dict)
                    print(f'positive number {pos_df.shape[0]}')
                    pos_output_path = 'customized_dataset/religion/only_random_positive.csv'
                    pos_df.to_csv(path + pos_output_path)

                    neg_dict = {'text':negative_texts}
                    neg_df = pd.DataFrame(neg_dict)
                    print(f'negative number {neg_df.shape[0]}')
                    neg_output_path = 'customized_dataset/religion/only_random_negative.csv'
                    neg_df.to_csv(path + neg_output_path)

                elif negative_sample_type == 'only_neg_query':
                    # only negative query
                    pos_dict = {'text':retrieved_texts}
                    neg_dict = {'text':n_retrieved_texts}
                    pos_df = pd.DataFrame(pos_dict)
                    neg_df = pd.DataFrame(neg_dict)

                    print(f'positive number {pos_df.shape[0]}')
                    pos_output_path = 'customized_dataset/religion/only_neg_query_positive.csv'
                    pos_df.to_csv(path + pos_output_path)

                    print(f'negative number {neg_df.shape[0]}')
                    neg_output_path = 'customized_dataset/religion/only_neg_query_negative.csv'
                    neg_df.to_csv(path + neg_output_path)                
                
                elif negative_sample_type == 'random_plus_neg_query':
                    neg_num = len(retrieved_texts) // 2 # 50% random sample
                    keys = random.sample(list(all_passages), neg_num)           
                    negative_texts = [all_passages[k][0] for k in keys]

                    cnt = 0
                    for j,text in enumerate(negative_texts):
                        if text in retrieved_texts or 'wiki:' in negative_texts[j]:
                            negative_texts.pop(j)
                            cnt += 1
                        elif text in n_retrieved_texts or 'wiki:' in negative_texts[j]:
                            negative_texts.pop(j)
                            cnt += 1
                    if cnt >= 1:
                        new_keys = random.sample(list(all_passages.values()), cnt)
                        negative_texts.extend(new_keys)

                    # random 50% sampling
                    neg_dict_random = {'text':negative_texts}
                    neg_df_random = pd.DataFrame(neg_dict_random)

                    # neg query 50% sampling
                    neg_dict = {'text':n_retrieved_texts}
                    neg_df = pd.DataFrame(neg_dict)
                    neg_df = neg_df.sample(n=neg_df.shape[0]//2, random_state=1234)

                    neg_df = pd.concat([neg_df,neg_df_random],axis=0)
                    print(f'negative number {neg_df.shape[0]}')
                    neg_output_path = 'customized_dataset/religion/random_plus_neg_query_negative.csv'
                    neg_df.to_csv(path + neg_output_path)

                    pos_dict = {'text':retrieved_texts}
                    pos_df = pd.DataFrame(pos_dict)
                    print(f'positive number {pos_df.shape[0]}')
                    pos_output_path = 'customized_dataset/religion/random_plus_neg_query_positive.csv'
                    pos_df.to_csv(path + pos_output_path)

            # South Korea keyword
            sk_total_queries = ['South Korea', 'K-pop', 'Kimchi', 'Seoul'] 

            sk_negative_queries = ['North Korea', 'Japan', 'China']

            sk_questions = []
            for query in sk_total_queries:
                sk_questions.append(query)

            n_sk_questions =[]
            for n_query in sk_negative_queries:
                n_sk_questions.append(n_query)

            retrieved_texts = constructing_dataset(sk_questions, 'customized_dataset')
            n_retrieved_texts = constructing_dataset(n_sk_questions, 'customized_dataset')

            for negative_sample_type in ['only_random', 'only_neg_query', 'random_plus_neg_query']:
                if negative_sample_type == 'only_random':
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
                    print(f'duplicated number {cnt}')

                    pos_dict = {'text':retrieved_texts}
                    pos_df = pd.DataFrame(pos_dict)
                    print(f'positive number {pos_df.shape[0]}')
                    pos_output_path = 'customized_dataset/south_korea/only_random_positive.csv'
                    pos_df.to_csv(path + pos_output_path)

                    neg_dict = {'text':negative_texts}
                    neg_df = pd.DataFrame(neg_dict)
                    print(f'negative number {neg_df.shape[0]}')
                    neg_output_path = 'customized_dataset/south_korea/only_random_negative.csv'
                    neg_df.to_csv(path + neg_output_path)

                elif negative_sample_type == 'only_neg_query':
                    # only negative query
                    pos_dict = {'text':retrieved_texts}
                    neg_dict = {'text':n_retrieved_texts}
                    pos_df = pd.DataFrame(pos_dict)
                    neg_df = pd.DataFrame(neg_dict)

                    print(f'positive number {pos_df.shape[0]}')
                    pos_output_path = 'customized_dataset/south_korea/only_neg_query_positive.csv'
                    pos_df.to_csv(path + pos_output_path)

                    print(f'negative number {neg_df.shape[0]}')
                    neg_output_path = 'customized_dataset/south_korea/only_neg_query_negative.csv'
                    neg_df.to_csv(path + neg_output_path)                
                
                elif negative_sample_type == 'random_plus_neg_query':
                    neg_num = len(retrieved_texts) // 2 # 50% random sample
                    keys = random.sample(list(all_passages), neg_num)           
                    negative_texts = [all_passages[k][0] for k in keys]

                    cnt = 0
                    for j,text in enumerate(negative_texts):
                        if text in retrieved_texts or 'wiki:' in negative_texts[j]:
                            negative_texts.pop(j)
                            cnt += 1
                        elif text in n_retrieved_texts or 'wiki:' in negative_texts[j]:
                            negative_texts.pop(j)
                            cnt += 1
                    if cnt >= 1:
                        new_keys = random.sample(list(all_passages.values()), cnt)
                        negative_texts.extend(new_keys)

                    # random 50% sampling
                    neg_dict_random = {'text':negative_texts}
                    neg_df_random = pd.DataFrame(neg_dict_random)

                    # neg query 50% sampling
                    neg_dict = {'text':n_retrieved_texts}
                    neg_df = pd.DataFrame(neg_dict)
                    neg_df = neg_df.sample(n=neg_df.shape[0]//2, random_state=1234)


                    neg_df = pd.concat([neg_df,neg_df_random],axis=0)
                    print(f'negative number {neg_df.shape[0]}')
                    neg_output_path = 'customized_dataset/south_korea/random_plus_neg_query_negative.csv'
                    neg_df.to_csv(path + neg_output_path)

                    pos_dict = {'text':retrieved_texts}
                    pos_df = pd.DataFrame(pos_dict)
                    print(f'positive number {pos_df.shape[0]}')
                    pos_output_path = 'customized_dataset/south_korea/random_plus_neg_query_positive.csv'
                    pos_df.to_csv(path + pos_output_path)

        elif TASK_NAME == 'factsnet':
            query_path = "/crawl/crawler/query_output"
            query_path_list = []

            for cat in os.listdir(query_path):
                if cat == 'general':
                    continue
                else:
                    sub_path = os.path.join(query_path,cat)

                    for cat_2 in os.listdir(sub_path):
                        edge_path = os.path.join(sub_path, cat_2)
                        for cat_3 in os.listdir(edge_path):
                            file_path = os.path.join(edge_path, cat_3)
                            query_path_list.append(file_path)
            
            from tqdm import tqdm
            for iiii, query_factsnet in tqdm(enumerate(query_path_list), leave=True):
                print('='*100)
                print(query_factsnet)
                print('='*100)

                # pos_output_path = path + "factsnet/" + f"{query_factsnet.split('/')[-3] + '/' + query_factsnet.split('/')[-1]}_positive.csv"
                # neg_output_path = path + "factsnet/" + f"{query_factsnet.split('/')[-3] + '/' + query_factsnet.split('/')[-1]}_negative.csv"
                
                output_path_folder = path + "factsnet/" + query_factsnet.split('/')[-3] + '/'
                os.makedirs(output_path_folder, exist_ok=True)

                pos_path_file = f"{query_factsnet.split('/')[-1]}_positive.csv"
                neg_path_file = f"{query_factsnet.split('/')[-1]}_negative.csv"

                pos_output_path = output_path_folder + pos_path_file
                neg_output_path = output_path_folder + neg_path_file

                ERROR_LIST = [
                    "building_dataset/factsnet/history/query_culture-facts.csv_positive.csv",
                    "building_dataset/factsnet/nature/query_cat-facts.csv_positive.csv",
                    "building_dataset/factsnet/nature/query_otter-facts.csv_positive.csv",
                    "building_dataset/factsnet/nature/query_bear-facts.csv_positive.csv",
                    "building_dataset/factsnet/nature/query_whale-facts.csv_positive.csv",
                    "building_dataset/factsnet/world/query_egypt-facts.csv_positive.csv"
                ]

                if pos_output_path in ERROR_LIST:
                    content_naming = pos_output_path.split('/')[-1].split('.csv')[0]
                    pos_output_path = "building_dataset/factsnet/error_update" + content_naming + "_positive.csv"
                    neg_output_path = "building_dataset/factsnet/error_update" + content_naming + "_negative.csv"

                    questions = []
                    total_queries = list(pd.read_csv(query_factsnet)['query'])

                    # negative_queries = []

                    for query in total_queries:
                        # question, answers = qa_sample.query, qa_sample.answers
                        questions.append(query)

                    retrieved_texts = constructing_dataset(questions, 'factsnet', 10000, random_seed)

                    # only random sample negative query
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
                    print(f'duplicated number {cnt}')

                    

                    pos_dict = {'text':retrieved_texts}
                    pos_df = pd.DataFrame(pos_dict)
                    print(f'positive number {pos_df.shape[0]}')
                    pos_df.to_csv(pos_output_path)

                    neg_dict = {'text':negative_texts}
                    neg_df = pd.DataFrame(neg_dict)
                    print(f'negative number {neg_df.shape[0]}')
                    neg_df.to_csv(neg_output_path)

                    print(iiii,'th pos/neg data build finish!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


if __name__ == "__main__":
    main()
