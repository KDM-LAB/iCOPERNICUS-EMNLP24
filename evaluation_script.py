import pandas as pd
from scipy.stats import entropy
import torch
from torchmetrics.text.infolm import InfoLM

from tqdm.auto import tqdm

# from egises.egises import Egises, PersevalParams

from collections import Counter

# distance measures
from rouge_score import rouge_scorer
from nltk.translate import meteor
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import warnings
import typer

import concurrent
import itertools
import os
import typing
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import csv
import os
import pickle
# import random
import re
import traceback

import nltk
# import typer
import ujson
from tqdm.auto import tqdm
# import utils
# from egises.egises import Summary, Document
import concurrent
import itertools
import os
import typing
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

pd.set_option('mode.chained_assignment',
              None)  # disable warning "A value is trying to be set on a copy of a slice from a DataFrame."

# from kutils import custom_sigmoid, write_scores_to_csv, divide_with_exception, calculate_minmax_proportion

from dataclasses import dataclass
import csv
import os.path
import traceback

import numpy as np
mod_name = os.environ.get("model")
mod_cat = os.environ.get("mod_cat")

def divide_with_exception(x, y):
    try:
        return x / y
    except ZeroDivisionError as err:
        return 0
    except Exception as error:
        print(traceback.format_exc())
    return 0


def calculate_minmax_proportion(x, y, epsilon=0):
    try:
        return (min(x, y)+epsilon) / (max(x, y)+epsilon)
    except ZeroDivisionError as err:
        # print(f"x:{x}, y:{y}")
        # print(traceback.format_exc())
        return 0
    except Exception as err:
        print(f"x:{x}, y:{y}")
        print(traceback.format_exc())

        return 0


def write_scores_to_csv(rows, fields=None, filename="scores.csv"):
    # print(type(rows))
    if not rows:
        return
    if fields:
        try:
            assert rows and len(rows[0]) == len(fields)
        except AssertionError as err:
            print(traceback.format_exc())
            print(f"fields: {fields}")
            print(rows[0])

            return
    if os.path.exists(filename):
        # append to existing file
        with open(filename, 'a') as f:
            # using csv.writer method from CSV package
            if fields:
                write = csv.writer(f)
            write.writerows(rows)
    else:  # create new file
        with open(filename, 'w') as f:
            # using csv.writer method from CSV package
            if fields:
                write = csv.writer(f)
            write.writerow(fields)
            write.writerows(rows)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def custom_sigmoid(x, alpha=4.0, beta=1.0):
    return 1 / (1 + ((10 ** alpha) * np.exp(-(10 ** beta) * x)))


@dataclass
class PersevalParams:
    ADP_alpha: float = 4.0
    ADP_beta: float = 1.0
    ACP_alpha: float = 4.0
    ACP_beta: float = 1.0
    EDP_alpha: float = 3.0
    EDP_beta: float = 1.0
    epsilon: float = 0.0000001


class Summary:
    def __init__(self, origin_model: str, doc_id, uid, summary_text):
        self.origin_model = origin_model
        self.doc_id = doc_id
        self.uid = uid
        self.summary_text = summary_text

    def __repr__(self):
        return f"Summary(origin_model='{self.origin_model}', doc_id='{self.doc_id}', uid='{self.uid}', summary_text='{self.summary_text}')"


class Document:
    def __init__(self, doc_id, doc_text, doc_summ, user_summaries: Iterable[Summary],
                 model_summaries: Iterable[Summary]):
        self.doc_id = doc_id
        self.doc_text = doc_text
        self.doc_summ = doc_summ
        self.user_summaries = user_summaries
        self.model_summaries = model_summaries
        self.summary_doc_distances = {}
        self.summary_summary_distances = {}  # deviation of user summaries
        self.summary_user_distances = {}  # accuracy of model summaries

    def __repr__(self):
        return f"Document(doc_id='{self.doc_id}', doc_summ='{self.doc_summ}', doc_text='{self.doc_text[:50]}...')"

    def populate_summary_doc_distances(self, measure: typing.Callable, max_workers=1):
        # check if summary_doc_distances in
        ukeys = [(self.doc_id, user_summary.origin_model, user_summary.uid) for user_summary in self.user_summaries]
        uargs = [(user_summary.summary_text, f"{self.doc_summ} {self.doc_text}") for user_summary in
                 self.user_summaries]
        mkeys = [(self.doc_id, model_summary.origin_model, model_summary.uid) for model_summary in
                 self.model_summaries]
        margs = [(model_summary.summary_text, f"{self.doc_summ} {self.doc_text}") for model_summary in
                 self.model_summaries]
        # print(f"uargs: {uargs}")
        if max_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Map the function to the data, distributing the workload among processes
                results = list(executor.map(measure, uargs + margs))
        else:
            results = list(map(measure, uargs + margs))
        self.summary_doc_distances = {k: v for k, v in zip(ukeys + mkeys, results)}

        # for user_summary in self.user_summaries:
        #     self.summary_doc_distances[(self.doc_id, user_summary.origin_model, user_summary.uid)] = measure((
        #         user_summary.summary_text, f"{self.doc_summ} {self.doc_text}"))
        #
        # for model_summary in self.model_summaries:
        #     self.summary_doc_distances[(self.doc_id, model_summary.origin_model, model_summary.uid)] = measure((
        #         model_summary.summary_text, f"{self.doc_summ} {self.doc_text}"))
        # # print(f"self.summary_doc_distances: {self.summary_doc_distances}")

    def populate_summary_summary_distances(self, measure: typing.Callable, max_workers=1):
        # calculate user_summary_summary_distances
        keys = [(self.doc_id, user_summary1.origin_model, user_summary1.uid, user_summary2.uid) for
                user_summary1, user_summary2 in itertools.permutations(self.user_summaries, 2)]
        m_keys = [(self.doc_id, model_summary1.origin_model, model_summary1.uid, model_summary2.uid) for
                  model_summary1, model_summary2 in itertools.permutations(self.model_summaries, 2)]
        su_keys = [(self.doc_id, model_summary.origin_model, model_summary.uid) for model_summary in
                   self.model_summaries]

        # get user generated summaries into a dictionary
        user_summary_dict = {(summary.doc_id, summary.uid): summary for summary in self.user_summaries}

        res_args = [(user_summary1.summary_text, user_summary2.summary_text) for user_summary1, user_summary2 in
                    itertools.permutations(self.user_summaries, 2)]
        m_res_args = [(model_summary1.summary_text, model_summary2.summary_text) for model_summary1, model_summary2 in
                      itertools.permutations(self.model_summaries, 2)]
        su_args = [(model_summary.summary_text, user_summary_dict[model_summary.doc_id, model_summary.uid].summary_text)
                   for model_summary in self.model_summaries]
        if max_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Map the function to the data, distributing the workload among processes
                results = list(executor.map(measure, res_args + m_res_args + su_args))
        else:
            results = list(map(measure, res_args + m_res_args + su_args))

        self.summary_summary_distances = {k: v for k, v in zip(keys + m_keys, results[:len(keys + m_keys)])}
        self.summary_user_distances = {k: v for k, v in zip(su_keys, results[len(keys + m_keys):])}


class Egises:
    def __init__(self, model_name, measure: typing.Callable, documents: Iterable[Document], score_directory="",
                 max_workers=1, debug_flag=True, version="v2"):
        self.model_name = model_name
        self.version = version
        if not score_directory:
            self.score_directory = f"{measure.__name__}/{model_name}"
        else:
            self.score_directory = score_directory
        if not os.path.exists(f"{self.score_directory}"):
            # create directory
            os.makedirs(f"{self.score_directory}")
            print(f"created directory: {self.score_directory}")
        else:
            print(f"directory already exists: {self.score_directory}")

        self.max_workers = max_workers
        self.debug_flag = debug_flag
        self.summary_doc_score_path = f"{self.score_directory}/sum_doc_distances.csv"
        self.summ_summ_score_path = f"{self.score_directory}/sum_sum_doc_distances.csv"
        self.sum_user_score_path = f"{self.score_directory}/sum_user_distances.csv"

        self.measure = measure
        self.documents = documents
        self.summary_doc_score_df = None
        self.summ_pair_score_df = None

    def populate_distances(self, simplified_flag=False):
        """
        :param simplified_flag: doesnt normalize scores based on doc distances
        :return:
        """
        last_seen, last_doc_processed = False, None
        processed_doc_ids = []
        if os.path.exists(self.summary_doc_score_path) and os.path.exists(self.summ_summ_score_path):
            summary_doc_score_df = pd.read_csv(self.summary_doc_score_path)
            # get unique doc_ids
            processed_doc_ids = list(summary_doc_score_df["doc_id"].unique())
        # populate document scores from where left off
        pbar = tqdm(total=3840, desc="Populating Distances")
        for document in self.documents:
            # find last doc_id in summary_doc_distances
            # avoid pouplating distances as hj distances already processed
            if self.measure.__name__ == "calculate_hj":
                break
            if document.doc_id in processed_doc_ids:
                pbar.update(1)
                continue
            summary_doc_tuples = []
            summ_pair_tuples = []
            summ_user_tuples = []
            document.populate_summary_doc_distances(self.measure, max_workers=self.max_workers)
            summary_doc_tuples.extend([(*k, v) for k, v in document.summary_doc_distances.items()])
            # print(f"self.summary_doc_tuples: {self.summary_doc_tuples}")
            document.populate_summary_summary_distances(self.measure, max_workers=self.max_workers)
            summ_pair_tuples.extend([(*k, v) for k, v in document.summary_summary_distances.items()])
            summ_user_tuples.extend([(*k, v) for k, v in document.summary_user_distances.items()])
            # print(f"self.summ_pair_tuples: {self.summ_pair_tuples}")
            # distance between summaries and documents
            write_scores_to_csv(summary_doc_tuples, fields=("doc_id", "origin_model", "uid", "score"),
                                filename=self.summary_doc_score_path)

            # distance between summaries
            write_scores_to_csv(summ_pair_tuples, fields=("doc_id", "origin_model", "uid1", "uid2", "score"),
                                filename=self.summ_summ_score_path)

            # distance between user/gold personalized summaries and model summaries
            write_scores_to_csv(summ_user_tuples, fields=("doc_id", "origin_model", "uid", "score"),
                                filename=self.sum_user_score_path)
            pbar.update(1)

        self.summary_doc_score_df = pd.read_csv(self.summary_doc_score_path)
        self.summ_pair_score_df = pd.read_csv(self.summ_summ_score_path)
        self.accuracy_df = pd.read_csv(self.sum_user_score_path)

        # calculate X,Y scores for all document,u1,u2  pairs
        self.user_X_df = self.get_user_model_X_scores(model_name="user")
        self.model_Y_df = self.get_user_model_X_scores(model_name=self.model_name)

        # create map of user_X_df[(doc_id,uid1,uid2)] to user_X_df["final_score"]
        user_X_df = self.user_X_df.set_index(["doc_id", "uid1", "uid2"])
        user_X_score_map = user_X_df.to_dict(orient="index")

        # calculate min/max on model_Y_df["final_score"] and user_X_score_map[(doc_id,uid1,uid2))]
        if not simplified_flag:
            self.model_Y_df["proportion"] = self.model_Y_df.apply(
                lambda x: calculate_minmax_proportion(x.final_score, user_X_score_map[
                    (x["doc_id"], x["uid1"], x["uid2"])]["final_score"], epsilon=0.00001), axis=1)
        else: # simplified version where propotion is not weighted
            self.model_Y_df["proportion"] = self.model_Y_df.apply(
                lambda x: calculate_minmax_proportion(x.score, user_X_score_map[
                    (x["doc_id"], x["uid1"], x["uid2"])]["score"], epsilon=0.00001), axis=1)

    def get_user_model_X_scores(self, model_name):
        usum_scores_df = self.summary_doc_score_df[self.summary_doc_score_df["origin_model"] == model_name]
        # TODO: rename to mpair_scores_df
        upair_scores_df = self.summ_pair_score_df[self.summ_pair_score_df["origin_model"] == model_name]

        usum_scores_df = usum_scores_df.set_index(["doc_id", "uid"])
        sum_doc_score_dict = {k: v["score"] for k, v in usum_scores_df.to_dict(orient="index").items()}

        # step2: get ratio of summary_summary_distance to summary_doc_distance
        # w(u_ij) = distance(ui,uj)/sum(distance(ui,doc))

        upair_scores_df["pair_score_weight"] = upair_scores_df.apply(
            lambda x: divide_with_exception(x["score"], sum_doc_score_dict[(x["doc_id"], x["uid1"])]), axis=1)

        # step 3: calculate softmax of pair_score_weight grouped by doc_id, uid1
        # softmax(w(u_ij)) = exp(w(u_ij))/sum(exp(w(u_il))) where l is all users who summarized doc i
        upair_scores_df["pair_score_weight_exp"] = upair_scores_df.apply(
            lambda x: np.exp(x["pair_score_weight"]),
            axis=1)
        upair_scores_df["pair_score_weight_exp_softmax"] = upair_scores_df.groupby(["doc_id", "uid1"])[
            "pair_score_weight_exp"].transform(lambda x: x / sum(x))

        upair_scores_df["final_score"] = upair_scores_df.apply(
            lambda x: round(x["pair_score_weight_exp_softmax"] * x["score"], 4), axis=1)
        # keep only doc_id, uid1, uid2, final_score
        final_df = upair_scores_df[["doc_id", "uid1", "uid2", "score", "final_score"]]
        return final_df

    def get_egises_score(self, sample_percentage=100):
        # sample doc_id,u1, u2 pairs from model_Y_df
        model_Y_df = self.model_Y_df.sample(frac=sample_percentage / 100)
        accuracy_dict = {(k[0], k[1]): v["score"] for k, v in self.accuracy_df.set_index(["doc_id", "uid"]).to_dict(
            orient="index").items()}

        # find mean of model_Y_df["final_score"] grouped by doc_id,uid1
        model_Y_df["doc_userwise_proportional_divergence"] = model_Y_df.groupby(["doc_id", "uid1"])[
            "proportion"].transform(
            lambda x: np.mean(x))

        # find mean of model_Y_df["doc_userwise_proportional_divergence"] grouped by doc_id
        model_Y_df["docwise_mean_proportion"] = model_Y_df.groupby(["doc_id"])[
            "doc_userwise_proportional_divergence"].transform(
            lambda x: np.mean(x))

        if self.debug_flag and sample_percentage == 100:
            # save model_Y_df to csv
            model_Y_df.to_csv(f"{self.score_directory}/model_Y_df_{self.version}.csv", index=False)

        # temporary df to calculate docwise_mean_proportion
        final_df = model_Y_df[["doc_id", "docwise_mean_proportion"]].drop_duplicates()

        # calculate mean of accuracy of model-user pairs
        doc_pairs = list(model_Y_df.groupby(["doc_id", "uid1"]).groups.keys())
        doc_pairs.extend(model_Y_df.groupby(["doc_id", "uid2"]).groups.keys())
        doc_pairs = list(set(doc_pairs))
        # print(doc_pairs[:2])
        # print(accuracy_dict.values())
        msum_accuracies = [accuracy_dict[pair] for pair in doc_pairs]
        mean_msum_accuracy = np.mean(msum_accuracies)

        # find mean of mean_proportion column
        return round(1 - final_df['docwise_mean_proportion'].mean(), 4), round(mean_msum_accuracy, 4)

    def calculate_edp(self, accuracy_df, perseval_params: PersevalParams) -> dict:
        # calculation of d_mean
        summ_user_mean_dict = accuracy_df.groupby(["doc_id", "origin_model"]).apply(
            lambda x: np.mean(x["score"])).to_dict()

        # calculation of d_min
        summ_user_min_dict = accuracy_df.groupby(["doc_id", "origin_model"]).apply(lambda x: min(x["score"])).to_dict()

        accuracy_df["d_min"] = accuracy_df.apply(lambda x: summ_user_min_dict[(x["doc_id"], x["origin_model"])],
                                                 axis=1)
        accuracy_df["d_mean"] = accuracy_df.apply(lambda x: summ_user_mean_dict[(x["doc_id"], x["origin_model"])],
                                                  axis=1)

        # calculate Accuracy Inconsistency Penalty(ACP)
        accuracy_df["pterm1"] = accuracy_df.apply(
            lambda x: ((x["score"] - x["d_min"]) / ((x["d_mean"] - x["d_min"]) + perseval_params.epsilon)), axis=1)
        # applied sigmoid to pterm1
        accuracy_df["ACP"] = accuracy_df.apply(
            lambda x: custom_sigmoid(x["pterm1"], alpha=perseval_params.ACP_alpha, beta=perseval_params.ACP_beta),
            axis=1)

        # calculate Accuracy Drop Penalty(ADP)
        accuracy_df["pterm2"] = accuracy_df.apply(
            lambda x: (x["d_min"] - 0) / (1 - x["d_min"] + perseval_params.epsilon), axis=1)
        accuracy_df["ADP"] = accuracy_df.apply(
            lambda x: custom_sigmoid(x["pterm2"], alpha=perseval_params.ADP_alpha, beta=perseval_params.ADP_beta),
            axis=1)

        # calculate Document Generalization Penalty(DGP)
        accuracy_df["DGP"] = accuracy_df.apply(lambda x: (x["ACP"] + x["ADP"]), axis=1)

        accuracy_df["EDP"] = accuracy_df.apply(
            lambda x: (1 - custom_sigmoid(x["DGP"], alpha=perseval_params.EDP_alpha, beta=perseval_params.EDP_beta)),
            axis=1)

        doc_user_edp_dict = accuracy_df.groupby(["doc_id", "uid"]).apply(lambda x: np.mean(x["EDP"])).to_dict()
        return doc_user_edp_dict

    def get_perseval_score(self, sample_percentage=100, perseval_params: PersevalParams = None):
        if not perseval_params:
            perseval_params = PersevalParams()
        # calculate_degress
        model_Y_df = self.model_Y_df.sample(frac=sample_percentage / 100)

        # for debug purpose
        if sample_percentage == 100 and self.debug_flag:
            model_Y_df.to_csv(f"{self.score_directory}/model_Y_df_perseval_df_{self.version}.csv", index=False)

        # find mean of model_Y_df["final_score"] grouped by doc_id,uid1
        model_Y_df["doc_userwise_proportional_divergence"] = model_Y_df.groupby(["doc_id", "uid1"])[
            "proportion"].transform(
            lambda x: np.mean(x))

        doc_user_degress_df = model_Y_df[["doc_id", "uid1", "doc_userwise_proportional_divergence"]].drop_duplicates()
        # get doc_id, uid1 pairs from doc_user_degress_df
        degress_pairs = list(doc_user_degress_df.groupby(["doc_id", "uid1"]).groups.keys())

        # pick records from accuracy_df where doc_id, uid in doc_user_degress_df
        accuracy_df = self.accuracy_df[
            self.accuracy_df.apply(lambda x: (x["doc_id"], x["uid"]) in degress_pairs, axis=1)]
        # calculate_edp based on sampled model_Y_df
        doc_user_edp_dict = self.calculate_edp(accuracy_df, perseval_params)

        try:
            assert len(doc_user_edp_dict) == len(doc_user_degress_df)
        except AssertionError as err:
            print(f"len(doc_user_edp_dict): {len(doc_user_edp_dict)}")
            print(f"len(doc_user_degress_df): {len(doc_user_degress_df)}")
            raise Exception("length of doc_user_edp_dict and doc_user_degress_df should be equal")

        doc_user_degress_df["edp"] = doc_user_degress_df.apply(
            lambda x: doc_user_edp_dict[(x["doc_id"], x["uid1"])], axis=1)
        doc_user_degress_df["perseval"] = doc_user_degress_df.apply(
            lambda x: x["doc_userwise_proportional_divergence"] * x["edp"], axis=1)
        doc_user_degress_df["docwise_perseval_proportion"] = doc_user_degress_df.groupby(["doc_id"])[
            "perseval"].transform(
            lambda x: np.mean(x))

        # for debug purpose
        if sample_percentage == 100 and self.debug_flag:
            doc_user_degress_df.to_csv(f"{self.score_directory}/doc_degress_perseval_df_{self.version}.csv",
                                       index=False)

        final_doc_df = doc_user_degress_df[["doc_id", "docwise_perseval_proportion"]].drop_duplicates()

        return round(final_doc_df['docwise_perseval_proportion'].mean(), 4), round(accuracy_df["score"].mean(), 4)
        # take docwise mean of perseval


DATA_SET_PATH = "dataset"


# def write_scores_to_csv(rows, fields=None, filename="scores.csv"):
#     # print(type(rows))
#     if fields:
#         try:
#             assert rows and len(rows[0]) == len(fields)
#         except AssertionError as err:
#             print(traceback.format_exc())
#             print(f"fields: {fields}")
#             print(rows[0])

#             return
#     if os.path.exists(filename):
#         # append to existing file
#         with open(filename, 'a') as f:
#             # using csv.writer method from CSV package
#             if fields:
#                 write = csv.writer(f)
#             write.writerows(rows)
#     else:
#         with open(filename, 'w') as f:
#             # using csv.writer method from CSV package
#             if fields:
#                 write = csv.writer(f)
#             write.writerow(fields)
#             write.writerows(rows)


def load_data(path):
    with open(path + '.pkl', 'rb') as file:
        var = pickle.load(file)
    return var


def _tokenize(text):
    stopwords = set(nltk.corpus.stopwords.words('english'))

    # wordnet lemmatizer
    lemmatizer = nltk.stem.WordNetLemmatizer()

    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation

    text = re.sub(r'[\d+]', '', text.lower())  # remove numerical values and convert to lower case

    tokens = nltk.word_tokenize(text)  # tokenization

    tokens = [token for token in tokens if token not in stopwords]  # removing stopwords

    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # lemmatization

    # my_string= " ".join(tokens)

    return tokens


def _tokenize_text(text):
    return " ".join(_tokenize(text))


def get_model_documents(model_name, filepath=f"{DATA_SET_PATH}/consolidated_data.jsonl", measure=""):
    with open(filepath, "r") as fpr:
        for line in fpr.readlines():
            line = line.strip()
            line = ujson.loads(line)
            doc_id, doc_text, doc_summ = line["doc_id"], line["doc_text"], line["doc_summ"]
            user_summaries = [Summary("user", doc_id, uid, model_summary_map["user"]) for uid, model_summary_map in
                              line["m_summary_dict"].items()]
            model_summaries = [Summary(model_name, doc_id, uid, model_summary_map[model_name]) for
                               uid, model_summary_map in line["m_summary_dict"].items()]
            yield Document(doc_id, doc_text, doc_summ, user_summaries, model_summaries)



DATA_SET_PATH = "dataset"
PERSONALIZED_MODELS = [mod_name]
NON_PERSONALIZED_MODELS_LIST = ()

warnings.filterwarnings('ignore')

app = typer.Typer()

CONSOLIDATED_FILEPATH = f"dataset/final_{mod_name}_tokenized_consolidated_data.jsonl"
SCORES_PATH = f"scores"

# load infoLM model only once
# TODO: load model based on function argument
device = 'cuda' if torch.cuda.is_available() else 'cpu'
infolm = InfoLM('google/bert_uncased_L-2_H-128_A-2', idf=False, device=device, alpha=1.0, beta=1.0,
                information_measure="ab_divergence",
                verbose=False)


def calculate_meteor(texts):
    text1, text2 = texts
    tokens1 = text1.split()
    tokens2 = text2.split()
    result = meteor([tokens1], tokens2)
    return round(result, 5)


def calculate_bleu(texts):
    text1, text2 = texts
    tokens1 = text1.split(" ")
    tokens2 = text2.split(" ")
    result = sentence_bleu([tokens1], tokens2)
    # print(round(result,5))
    return round(result, 5)


def calculate_rougeL(texts):
    text1, text2 = texts
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    result = scorer.score(text1, text2)["rougeL"].fmeasure
    # print(round(result,5))
    return round(result, 5)


def calculate_rougeSU4(texts):
    candidate, reference = texts
    candidate = candidate.split(" ")
    reference = reference.split(" ")
    # Calculate skip-bigram matches upto 5 gram
    for i in range(5):
        candidate_ngrams = [tuple(candidate[j:j + i + 1]) for j in range(len(candidate) - i)]
        reference_ngrams = [tuple(reference[j:j + i + 1]) for j in range(len(reference) - i)]
    # Calculate the number of skip-n-gram matches
    match_count = sum((Counter(candidate_ngrams) & Counter(reference_ngrams)).values())
    # Calculate the number of skip-ngrams in the candidate and reference summaries
    candidate_bigram_count = len(candidate_ngrams)
    reference_bigram_count = len(reference_ngrams)
    # Calculate precision, recall, and F-measure
    precision = match_count / candidate_bigram_count if candidate_bigram_count > 0 else 0.0
    recall = match_count / reference_bigram_count if reference_bigram_count > 0 else 0.0
    beta = 1  # Set beta to 1 for ROUGE-SU4
    f_measure = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall) if (
                                                                                                   precision + recall) > 0 else 0.0
    # return precision, recall, f_measure
    return round(f_measure, 5)


def _text2distribution(text: list, common_vocab: set):
    """
    Calculate the probability distribution of words in the given text with respect to the common vocabulary.

    Parameters:
    - text: List of words.
    - common_vocab: Common vocabulary list.

    Returns:
    - prob_dist: Probability distribution represented as a numpy array.
    """
    word_counts = Counter(text)
    total_words = len(text)

    # Initialize probability distribution with zeros
    prob_dist = np.zeros(len(common_vocab))
    if total_words == 0:
        return prob_dist
    # Populate the probability distribution based on the common vocabulary
    for i, word in enumerate(common_vocab):
        prob_dist[i] = word_counts[word] / total_words

    return prob_dist


def calculate_JSD(texts):
    # create common vocab
    tokens_1, tokens_2 = [text.split() for text in texts]
    common_vocab = set(tokens_1).union(set(tokens_2))

    # calculate probability distributions
    p_dist = _text2distribution(tokens_1, common_vocab)
    q_dist = _text2distribution(tokens_2, common_vocab)

    m_dist = 0.5 * (p_dist + q_dist)

    # Calculate Kullback-Leibler divergences
    kl_p = entropy(p_dist, m_dist, base=2)
    kl_q = entropy(q_dist, m_dist, base=2)

    # Calculate Jensen-Shannon Divergence
    jsd_value = 0.5 * (kl_p + kl_q)
    jsd_value = round(jsd_value, 4)
    return jsd_value


def calculate_infoLM(texts: list):
    pred, target = texts
    score = infolm([pred], [target]).item()
    return round(score, 5)


from evaluate import load

bertscore = load("bertscore")


def calculate_bert_score(texts: list):
    pred, target = texts
    score = bertscore.compute(predictions=[pred], references=[target], lang="en", model_type="distilbert-base-uncased",
                              device='cuda')
    return score['f1'][0]


def calculate_hj(texts: list):
    raise Exception("Not implemented")


@app.command()
def populate_distances(model_name: str, distance_measure: str, max_workers: int = 1):
    """
    model_name: one of PERSONALIZED_MODELS or NON_PERSONALIZED_MODELS_LIST
    distance_measure: one of meteor, bleu, rougeL, rougeSU4, infoLM, jsd
    max_workers: number of workers to use for multiprocessing
    """
    measure_dict = {
        "meteor": calculate_meteor,
        "bleu": calculate_bleu,
        "rougeL": calculate_rougeL,
        "rougeSU4": calculate_rougeSU4,
        "infoLM": calculate_infoLM,
        "JSD": calculate_JSD,
        "bert_score": calculate_bert_score,
        "hj": calculate_hj
    }

    if distance_measure == "infoLM" and max_workers > 1:
        print(f"setting max_workers to 1 for infoLM")
        max_workers = 1

    try:
        assert distance_measure in measure_dict.keys()
        measure = measure_dict[distance_measure]
    except AssertionError as err:
        print(f"measure should be one of {measure_dict.keys()}")
        return
    eg = Egises(model_name=model_name, measure=measure,
                documents=get_model_documents(model_name, CONSOLIDATED_FILEPATH),
                score_directory=f"{SCORES_PATH}/{measure.__name__}/{mod_cat}",
                max_workers=max_workers)
    eg.populate_distances()


@app.command()
def generate_scores(distance_measure: str,version: str, sampling_freq: int = 10, max_workers: int = 1,simplified_flag: bool = False):
    """
    distance_measure: one of meteor, bleu, rougeL, rougeSU4, infoLM, jsd
    sampling_freq: sampling frequency for percentage less than 100
    max_workers: number of workers to use for multiprocessing
    version: generate suffixed scores files to avoid overwriting
    saves scores in scores/distance_measure/egises_scores_version.csv
    """
    print(mod_name)
    print(mod_cat)
    measure_dict = {
        "meteor": calculate_meteor,
        "bleu": calculate_bleu,
        "rougeL": calculate_rougeL,
        "rougeSU4": calculate_rougeSU4,
        "infoLM": calculate_infoLM,
        "JSD": calculate_JSD,
        "bert_score": calculate_bert_score,
        "hj": calculate_hj
    }

    if distance_measure == "infoLM" and max_workers > 1:
        print(f"setting max_workers to 1 for infoLM")
        max_workers = 1

    try:
        assert distance_measure in measure_dict.keys()
        measure = measure_dict[distance_measure]
    except AssertionError as err:
        print(f"measure should be one of {measure_dict.keys()}")
        return
    # measure = calculate_meteor
    for model_name in tqdm(PERSONALIZED_MODELS):
        distance_directory = f"{SCORES_PATH}/{measure.__name__}/{mod_cat}"
        # for model_name in tqdm([*PERSONALIZED_MODELS]):
        model_egises_tuple, model_accuracy_tuple = [model_name], [model_name]
        eg = Egises(model_name=model_name, measure=measure,
                    documents=get_model_documents(model_name, CONSOLIDATED_FILEPATH),
                    score_directory=distance_directory, max_workers=max_workers, version=version)
        eg.populate_distances(simplified_flag=simplified_flag)
        for sample_percentage in range(100, 10, -20):
            print(f"calculating for {model_name} with sample percentage {sample_percentage}")
            if sample_percentage == 100:
                eg_score, accuracy_score = eg.get_egises_score(sample_percentage=sample_percentage)
                print(f"eg_score: {eg_score}, accuracy_score: {accuracy_score}")
            else:
                # for sample percentage less than 100, calculate score 10 times and take mean
                eg_scores = []
                accuracy_scores = []
                pbar = tqdm(range(sampling_freq))
                for i in range(sampling_freq):
                    eg_score, accuracy_score = eg.get_egises_score(sample_percentage=sample_percentage)
                    eg_scores.append(eg_score)
                    accuracy_scores.append(accuracy_score)
                    pbar.update(1)
                pbar.close()
                eg_score = round(np.mean(eg_scores), 4)
                accuracy_score = round(np.mean(accuracy_scores), 4)
                print(f"eg_score: {eg_score}, accuracy_score: {accuracy_score}")
            model_egises_tuple.append(eg_score)
            model_accuracy_tuple.append(accuracy_score)

        std = np.std(model_egises_tuple[1:])
        # calculate vaBaseExceptionriance of model_tuple[1:]
        var = np.var(model_egises_tuple[1:])
        model_egises_tuple.append(round(std, 4))
        model_egises_tuple.append(var)

        std = np.std(model_accuracy_tuple[1:])
        # calculate vaBaseExceptionriance of model_tuple[1:]
        var = np.var(model_accuracy_tuple[1:])
        model_accuracy_tuple.append(round(std, 4))
        model_accuracy_tuple.append(var)

        print(f"model_egises_tuple: {model_egises_tuple}")
        print(f"model_accuracy_tuple: {model_accuracy_tuple}")
        write_scores_to_csv([model_egises_tuple],
                                  fields=["models", *list(range(100, 10, -20)), "bias", "variance"],
                                  filename=f"{SCORES_PATH}/{measure.__name__}/egises_scores_{version}.csv")

        write_scores_to_csv([model_accuracy_tuple],
                                  fields=["models", *list(range(100, 10, -20)), "bias", "variance"],
                                  filename=f"{SCORES_PATH}/{measure.__name__}/accuracy_scores_{version}.csv")


@app.command()
def generate_perseval_scores(distance_measure: str, version:str, sampling_freq: int = 10, max_workers: int = 1,
                             simplified_flag: bool = False):
    """
    distance_measure: one of meteor, bleu, rougeL, rougeSU4, infoLM, jsd
    model_name: sampling frequency for percentage less than 100
    max_workers: number of workers to use for multiprocessing
    simplified_flag: if True, calculate proportions without using doc based normalization
    version: generate suffixed scores files to avoid overwriting
    saves scores in scores/distance_measure/perseval_scores_version.csv
    """
    measure_dict = {
        "meteor": calculate_meteor,
        "bleu": calculate_bleu,
        "rougeL": calculate_rougeL,
        "rougeSU4": calculate_rougeSU4,
        "infoLM": calculate_infoLM,
        "JSD": calculate_JSD,
        "bert_score": calculate_bert_score,
        "hj": calculate_hj,

    }

    if distance_measure == "infoLM" and max_workers > 1:
        print(f"setting max_workers to 1 for infoLM")
        max_workers = 1

    try:
        assert distance_measure in measure_dict.keys()
        measure = measure_dict[distance_measure]
    except AssertionError as err:
        print(f"measure should be one of {measure_dict.keys()}")
        return
    # measure = calculate_meteor
    for model_name in tqdm(PERSONALIZED_MODELS):
        distance_directory = f"{SCORES_PATH}/{measure.__name__}/{mod_cat}"
        # for model_name in tqdm([*PERSONALIZED_MODELS]):
        model_perseval_tuple, model_accuracy_tuple = [model_name], [model_name]
        eg = Egises(model_name=model_name, measure=measure,
                    documents=get_model_documents(model_name, CONSOLIDATED_FILEPATH),
                    score_directory=distance_directory, max_workers=max_workers, version=version)
        eg.populate_distances(simplified_flag=simplified_flag)

        perseval_params = PersevalParams()
        print(f"calculating for {model_name} with perseval params {perseval_params}")
        for sample_percentage in range(100, 10, -20):
            print(f"sample percentage:{sample_percentage}")
            if sample_percentage == 100:
                perseval_score, accuracy_score = eg.get_perseval_score(sample_percentage=sample_percentage,
                                                                       perseval_params=perseval_params)
                print(f"perseval_score@{sample_percentage}%: {perseval_score}, accuracy_score: {accuracy_score}")
            else:
                # for sample percentage less than 100, calculate score 10 times and take mean
                perseval_scores = []
                accuracy_scores = []
                pbar = tqdm(range(sampling_freq))
                for i in range(sampling_freq):
                    perseval_score, accuracy_score = eg.get_perseval_score(sample_percentage=sample_percentage,
                                                                           perseval_params=perseval_params)
                    perseval_scores.append(perseval_score)
                    accuracy_scores.append(accuracy_score)
                    pbar.update(1)
                pbar.close()
                perseval_score = round(np.mean(perseval_scores), 4)
                accuracy_score = round(np.mean(accuracy_scores), 4)
                print(f"perseval_score@{sample_percentage}%: {perseval_score}, accuracy_score: {accuracy_score}")
            model_perseval_tuple.append(perseval_score)
            model_accuracy_tuple.append(accuracy_score)

        std = np.std(model_perseval_tuple[1:])
        # calculate vaBaseExceptionriance of model_tuple[1:]
        var = np.var(model_perseval_tuple[1:])
        model_perseval_tuple.append(round(std, 4))
        model_perseval_tuple.append(var)

        std = np.std(model_accuracy_tuple[1:])
        # calculate vaBaseExceptionriance of model_tuple[1:]
        var = np.var(model_accuracy_tuple[1:])
        model_accuracy_tuple.append(round(std, 4))
        model_accuracy_tuple.append(var)

        print(f"model_perseval_tuple: {model_perseval_tuple}")
        print(f"model_accuracy_tuple: {model_accuracy_tuple}")
        write_scores_to_csv([model_perseval_tuple],
                                  fields=["models", *list(range(100, 10, -20)), "bias", "variance"],
                                  filename=f"{SCORES_PATH}/{measure.__name__}/perseval_scores_{version}_simp_{simplified_flag}.csv")

        write_scores_to_csv([model_accuracy_tuple],
                                  fields=["models", *list(range(100, 10, -20)), "bias", "variance"],
                                  filename=f"{SCORES_PATH}/{measure.__name__}/perseval_accuracy_scores_{version}_simp_{simplified_flag}.csv")


def _get_measure_df(version: str,measure: str = "", p_measure: str = ""):
    version = f"_{version}" if version else ""
    csv_file = f"{SCORES_PATH}/calculate_{measure}/{p_measure}_scores{version}.csv"
    df = pd.read_csv(csv_file)
    return df


def _get_measure_scores(version: str ,measure: str = "", p_measure: str = ""):
    if p_measure == "degress":
        p_measure = "egises"
        df = _get_measure_df(measure, p_measure, version)
        df = df[["models", "100"]]
        df = df.set_index(["models"])
        measure_dict = df.to_dict(orient="index")
        measure_dict = {item[0]: 1 - item[1]["100"] for item in measure_dict.items()}
        return measure_dict

    df = _get_measure_df(measure, p_measure, version)
    df = df[["models", "100"]]
    df = df.set_index(["models"])
    measure_dict = df.to_dict(orient="index")
    measure_dict = {item[0]: item[1]["100"] for item in measure_dict.items()}
    return measure_dict


def _get_correlation_from_model_dict(model1: dict, model2: dict):
    sorted_measure1_dict = dict(sorted(model1.items(), key=lambda item: item[1]))

    # print(f"sorted_measure1_dict: {sorted_measure1_dict}")
    measure1_list = list(sorted_measure1_dict.values())
    measure2_list = [model2[model] for model in sorted_measure1_dict.keys()]
    measure1_list = pd.Series(measure1_list)
    measure2_list = pd.Series(measure2_list)

    # Calculating correlation
    corr_types = ['pearson', 'kendall', 'spearman']
    corr_dict = {corr_type: round(measure1_list.corr(measure2_list, method=corr_type), 5) for corr_type in corr_types}
    return corr_dict


@app.command()
def calculate_correlation(dmeasure_1: str, dmeasure_2: str,m2_version:str,m1_version:str,pmeasure1: str = "perseval", pmeasure2: str = "perseval"):
    """
    dmeasure_1: one of meteor, bleu, rougeL, rougeSU4, infoLM, jsd
    dmeasure_2: one of meteor, bleu, rougeL, rougeSU4, infoLM, jsd, hj
    pmeasure1: one of egises, perseval, degress, perseval_accuracy
    pmeasure2: one of egises, perseval, degress, perseval_accuracy
    """
    assert pmeasure1 in ["egises", "perseval", "perseval_accuracy", "degress"]
    assert pmeasure2 in ["egises", "perseval", "perseval_accuracy", "degress"]
    assert dmeasure_1 in ["meteor", "bleu", "rougeL", "rougeSU4", "infoLM", "JSD", "hj", "bert_score"]
    assert dmeasure_2 in ["meteor", "bleu", "rougeL", "rougeSU4", "infoLM", "JSD", "hj", "bert_score"]

    measure1_dict = _get_measure_scores(measure=dmeasure_1, p_measure=pmeasure1, version=m1_version)
    measure2_dict = _get_measure_scores(measure=dmeasure_2, p_measure=pmeasure2, version=m2_version)
    corr_dict = _get_correlation_from_model_dict(measure1_dict, measure2_dict)
    return corr_dict


def _calculate_borda_consensus(rank1: list, rank2: list) -> dict:
    """
    rank1: list of models in order of rank
    rank2: list of models in order of rank
    """
    assert len(rank1) == len(rank2)
    n = len(rank1)
    rank1_dict = {model: n - i for i, model in enumerate(rank1)}
    rank2_dict = {model: n - i for i, model in enumerate(rank2)}
    borda_dict = {}
    for model in rank1_dict.keys():
        borda_dict[model] = rank1_dict[model] + rank2_dict[model]
    borda_dict = dict(sorted(borda_dict.items(), key=lambda item: item[1]))
    return borda_dict


@app.command()
def get_borda_scores(dmeasure_1: str = "infoLM", dmeasure_2: str = "rougeL", p1_measure: str = "perseval",
                     p2_measure: str = "perseval_accuracy", m1_version="v2",
                     m2_version="v2"):
    """
    dmeasure_1: one of meteor, bleu, rougeL, rougeSU4, infoLM, jsd
    dmeasure_2: one of meteor, bleu, rougeL, rougeSU4, infoLM, jsd, hj
    p_measure: one of egises, perseval_accuracy
    """
    assert p1_measure in ["egises", "perseval", "perseval_accuracy"]
    assert p2_measure in ["egises", "perseval", "perseval_accuracy"]
    assert dmeasure_1 in ["meteor", "bleu", "rougeL", "rougeSU4", "infoLM", "jsd", "hj"]
    assert dmeasure_2 in ["meteor", "bleu", "rougeL", "rougeSU4", "infoLM", "jsd", "hj"]

    dmeasure_1_dict = _get_measure_scores(measure=dmeasure_1, p_measure=p1_measure, version=m1_version)
    dmeasure_2_dict = _get_measure_scores(measure=dmeasure_2, p_measure=p2_measure, version=m2_version)

    sorted_dmeasure_1_dict = dict(sorted(dmeasure_1_dict.items(), key=lambda item: - item[1]))
    sorted_dmeasure_2_dict = dict(sorted(dmeasure_2_dict.items(), key=lambda item: - item[1]))

    rank1 = list(sorted_dmeasure_1_dict.keys())
    rank2 = list(sorted_dmeasure_2_dict.keys())
    borda_dict = _calculate_borda_consensus(rank1, rank2)
    # print(f"borda_dict: {borda_dict}")
    return borda_dict


if __name__ == "__main__":
    app()
    # for acc_measure in ["bleu", "meteor", "rougeL", "rougeSU4", "infoLM"]:
    #     bk = get_borda_scores(dmeasure_1="hj", dmeasure_2=acc_measure, p1_measure="perseval",
    #                           p2_measure="perseval_accuracy", m1_version="v2", m2_version="v2")
    #     # print(f"bk:{bk}")
    #     bk = dict(sorted(bk.items(), key=lambda item: item[1]))
    #     bk = {key: i for i, key in enumerate(bk.keys(), 1)}
    #     # print(f"bk:{bk}")
    #     hj_scores = _get_measure_scores(measure="hj", p_measure="perseval", version="v2")
    #     hj_scores = dict(sorted(hj_scores.items(), key=lambda item: item[1]))
    #     # print(f"hj_scores:{hj_scores}")
    #     hj_scores = {key: i for i, key in enumerate(hj_scores.keys(), 1)}
    #     # print(f"hj_scores:{hj_scores}")
    #     corr_dict = _get_correlation_from_model_dict(bk, hj_scores)
    #     print(f"corr_dict_{acc_measure}:{corr_dict}")
    # hj_scores =
    # bk = _calculate_borda_consensus(["a", "c", "b"], ["a", "b", "c"])
    # print(bk)

    # measure_dict = {
    #     "rougeL": calculate_rougeL,
    #     "rougeSU4": calculate_rougeSU4,
    #     "meteor": calculate_meteor,
    #     "bleu": calculate_bleu,
    #     "infoLM": calculate_infoLM,
    #     "JSD": calculate_JSD,
    #     "bert_score": calculate_bert_score,
    # }
    # measure_list = ["rougeL", "rougeSU4", "meteor", "bleu", "infoLM", "bert_score", "JSD"]
    # measure_pearson_list, spearman_list, kendal_list = [], [], []
    # for measure in measure_dict.keys():
    #     print(f"measure: {measure}")
    #     corr_dict = calculate_correlation(dmeasure_1=measure, dmeasure_2=measure, pmeasure1="degress", pmeasure2="perseval", m1_version="sfinal",
    #                       m2_version="final")
    #     for corr_method in corr_dict.keys():
    #         # print(f"{measure}_hj_{corr_method}:{corr_dict[corr_method]}")
    #         print(f"{corr_dict[corr_method]}")
    #     print(f"*" * 50)
    # print(f"{measure}_perseval_{measure}-hj_perseval_rouge:{corr_dict}")
