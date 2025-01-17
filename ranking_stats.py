import os

import config
from config import current_prompt as cp
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM
import torch
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

warnings.filterwarnings("ignore")

# if not os.path.exists(f"feature_data_{cp}.csv"):
#     print("Part 1 started")
#     file_path = f'/lv_local/home/niv.b/content_modification_code-master/Results/RankedLists/LambdaMART{cp}'
#
#     columns = ['query_id', 'Q0', 'docno', 'rank', 'score', 'method']
#
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#
#     data = []
#
#     for line in lines:
#         values = line.strip().split()
#         query_id, _, document_id, rank, score, method = values
#         query_id_padded = "{:03}".format(int(query_id))
#         data.append([query_id_padded, 'Q0', document_id, int(rank), float(score), method])
#
#     df = pd.DataFrame(data, columns=columns).drop(["Q0", "method"], axis=1)
#
#     ### Features preprocess ####
#     folder_path = f'/lv_local/home/niv.b/content_modification_code-master/Results/Features/{cp}'
#
#     # Iterate over files in the folder
#     for filename in tqdm(os.listdir(folder_path), desc="Processing files", total=len(os.listdir(folder_path)),
#                          miniters=100):
#         file_path = os.path.join(folder_path, filename)
#         feat, qid = file_path.split('/')[-1].split("_")
#         if feat not in df.columns:
#             df[feat] = np.nan
#         with open(file_path, 'r') as file:
#             lines = file.readlines()
#             # Process each line and extract the values for each column
#             for line in lines:
#                 values = line.strip().split()
#                 try:
#                     docno, score = values
#                     df.loc[df['docno'] == docno, feat] = score
#                 except:
#                     try:
#                         docno, sum_, min_, max_, mean_, var_ = values
#                         df.loc[df['docno'] == docno, feat] = mean_
#                     except:
#                         x = 1
#
#     df.to_csv(f"feature_data_{cp}.csv", index=False)
#     print("Part 1 ended")

#### new features ####
if not os.path.exists(f"feature_data_{cp}_new.csv"):
    print("Part 2 started")
    df = pd.read_csv(f"feature_data_{cp}.csv").rename({"query_id": "query_id_new"}, axis=1)
    df = df[[col for col in df.columns if not col.startswith("doc")] + ['docno']]

    # df.query_id_new = df.query_id_new.astype(int)
    # df = pd.read_csv(f"feature_data_{cp}.csv").drop("rank", axis=1)

    # TODO: assuming we run asrcqrels only!
    if 'qrels' in cp:
        df['query_id_new'] = df['query_id_new'].astype(str)
        df[["round_no", "query_id", "username", "creator"]] = df["query_id_new"].apply(
            lambda x: pd.Series([x[0], x[1:4], x[4:6], x[6:]]))
        df = df[~df.docno.str.contains("creator")]


    else:
        # df[["round_no", "query_id", "username", "creator"]] = df["docno"].apply(lambda x: pd.Series(x.split("-")))
        df[["round_no", "query_id", "username", "creator"]] = df["docno"].apply(
            lambda x: pd.Series(x.split("-")[:]))  # TOMMY DATA
        # df["creator"] = "creator"

    maximal_epoch, minimal_epoch = df.round_no.max(), df.round_no.min()
    df.round_no, df.query_id = df.round_no.astype(int), df.query_id.astype(int)

    if config.using_e5:
        greg_data = pd.read_csv("tommy_data.csv")[['round_no', 'query_id', 'username', 'position']].rename(
            {"position": "original_position"}, axis=1)
    else:
        greg_data = pd.read_csv("greg_data.csv")[['round_no', 'query_id', 'username', 'position']].rename(
            {"position": "original_position"}, axis=1)

    greg_data.round_no, greg_data.query_id, greg_data.username = greg_data.round_no.astype(
        int), greg_data.query_id.astype(
        int), greg_data.username.astype(str)

    # TOMMY
    # df.round_no, df.query_id, df.username = df.round_no.astype(
    #     int), df.query_id.astype(
    #     int), df.username.astype(str)
    df = df.merge(greg_data, on=['round_no', 'query_id', 'username'], how='left', suffixes=['', '_x'])  # TOMMY

    df = df.merge(greg_data, right_on=['round_no', 'query_id', 'username'], left_on=['round_no', 'query_id', 'creator'],
                  how='left', suffixes=['', '_y'])

    for col in ["username", "original_position"]:
        for suffix in ["_x", "_y"]:
            if f"{col}{suffix}" in df.columns:
                df[col] = df[col].fillna(df[f"{col}{suffix}"])
                df = df.drop([f"{col}{suffix}"], axis=1)

    prev_round = []
    for _, row in greg_data[greg_data.round_no == int(minimal_epoch) - 1].iterrows():
        query_str = '0' + str(row.query_id)
        docno = f"0{int(minimal_epoch) - 1}-{query_str[-3:]}-{row.username}-creator"
        prev_round.append(
            {"round_no": row.round_no, "query_id": row.query_id, "creator": "creator", "username": row.username,
             "docno": docno, "original_position": row.original_position})
    prev_df = pd.DataFrame(prev_round)
    df = pd.concat([prev_df, df]).reset_index(drop=True)
    df.round_no, df.query_id = df.round_no.astype(int), df.query_id.astype(int)
    # df.set_index("docno", inplace=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows", miniters=100):
        try:
            if not int(minimal_epoch) <= row.round_no <= int(maximal_epoch):
                df.loc[idx, "previous_docno"] = np.nan
                continue

            if row.creator != "creator":  # bot docs
                prev_docno = df.index[
                    (df.round_no == row.round_no - 1) & (df.query_id == row.query_id) & (df.creator == "creator") & (
                            df.username == row.creator)].tolist()[0]
                # TOMMY
                orig_docno = df.index[
                    (df.round_no == row.round_no) & (df.query_id == row.query_id) & (df.creator == "creator") & (
                            df.username == row.creator)].tolist()[0]
                df.at[idx, "original_position"] = df[df.index == orig_docno].original_position.values[0]
            else:  # student docs
                prev_docno = df.index[
                    (df.round_no == row.round_no - 1) & (df.query_id == row.query_id) & (df.creator == "creator") & (
                            df.username == row.username)].tolist()[0]

            # Set the 'previous_docno' and 'previous_pos' for the current row based on the found 'previous_docno'
            df.loc[idx, "previous_docno"] = prev_docno
            df.loc[idx, "previous_docno_str"] = df.iloc[prev_docno]['docno']
            df.loc[idx, "previous_pos"] = df[df.index == prev_docno].original_position.values[0]
        except:
            continue

    if "username_x" in df.columns:
        df["username"] = df["username"].fillna(df["username_x"])
        df = df.drop(["username_x"], axis=1)
    if "username_y" in df.columns:
        df["username"] = df["username"].fillna(df["username_y"])
        df = df.drop(["username_y"], axis=1)

    df.to_csv(f"feature_data_{cp}_new.csv", index=False)

    # # iterating over groups to get their ranks (done if queries are the normal 31 only)
    # # Grouping the DataFrame by 'round_no' and 'query_id' for further processing
    # df_gb = df.groupby(["query_id_new"])
    #
    # # Looping through each group
    # for group_name, df_group in tqdm(df_gb):
    #     # Splitting the group into 'non_bots' (where 'creator' is "creator") and 'bots' (where 'creator' is not "creator")
    #     if int(str(group_name[0])[0]) not in list(range(int(minimal_epoch), int(maximal_epoch) + 1)):
    #         continue
    #     non_bots = df_group[df_group.creator == "creator"]
    #     bots = df_group[df_group.creator != "creator"]
    #
    #     # Looping through each row in 'bots' group to calculate 'current_pos' for each bot
    #     for idx, row in bots.iterrows():
    #         # Concatenating the 'non_bots' group and the current bot row to create a comparison DataFrame
    #         comp_df = pd.concat([non_bots, row.to_frame().T])
    #
    #         # Removing the current bot from the comparison DataFrame
    #         comp_df = comp_df[(comp_df.username != row.creator) | (comp_df.creator == row.username)]
    #
    #         # Calculating ranks based on the 'score' column in the comparison DataFrame
    #         comp_df['calc_rank'] = comp_df['score'].rank(ascending=False, method='dense')
    #
    #         # Setting the 'current_pos' for the current bot in the main DataFrame
    #         assert comp_df.loc[idx, "calc_rank"] % 1 == 0
    #         df.loc[idx, "current_pos"] = comp_df.loc[idx, "calc_rank"]
    #         if comp_df.loc[idx, "calc_rank"].astype(int) != df.loc[idx, "rank"].astype(int):
    #             x = 1

    df = df.rename({"rank": "current_pos"}, axis=1)

    # df.loc[df.creator == 'creator', 'current_pos'] = df.loc[df.creator == 'creator', "original_position"]

    # ORIGINAL MISTAKE

    # df['pos_diff'] = df.apply(lambda row: max(int(row['current_pos'] - row['previous_pos']) * -1, 0) if pd.notna(
    #     row['current_pos']) and pd.notna(row['previous_pos']) else np.nan, axis=1)
    # df['scaled_pos_diff'] = df.apply(
    #     lambda row: row['pos_diff'] / (row['previous_pos'] - 1) if pd.notna(row['previous_pos']) and row[
    #         'previous_pos'] != 1 else np.nan, axis=1)

    df['pos_diff'] = df['previous_pos'] - df['current_pos']


    def scaled_difference(row):
        if row['pos_diff'] > 0:  # Promotion
            return row['pos_diff'] / (row['previous_pos'] - 1)
        elif row['pos_diff'] < 0:  # Demotion
            try:
                return row['pos_diff'] / (4 - row['previous_pos'])
            except:
                return 0
        else:
            return 0  # No change


    df['scaled_pos_diff'] = df.apply(scaled_difference, axis=1)

    # Calculating 'scaled_orig_pos_diff' based on 'orig_pos_diff', considering some conditions
    # df['orig_pos_diff'] = df.apply(
    #     lambda row: max(int(row['current_pos'] - row['original_position']) * -1, 0) if pd.notna(
    #         row['current_pos']) and pd.notna(row['original_position']) else np.nan, axis=1)
    #
    # df['scaled_orig_pos_diff'] = df.apply(
    #     lambda row: row['orig_pos_diff'] / (row['original_position'] - 1) if pd.notna(row['original_position']) and row[
    #         'original_position'] != 1 else np.nan, axis=1)

    df['orig_pos_diff'] = df['original_position'] - df['current_pos']


    def scaled_orig_difference(row):
        if row['orig_pos_diff'] > 0:  # Promotion
            return row['orig_pos_diff'] / (row['original_position'] - 1)
        elif row['orig_pos_diff'] < 0:  # Demotion
            try:
                return row['orig_pos_diff'] / (4 - row['original_position'])
            except:
                return 0
        else:
            return 0  # No change


    df['scaled_orig_pos_diff'] = df.apply(scaled_orig_difference, axis=1)

    df = df[df.round_no != 1]
    df = df[df.score.notna()]

    # TODO: remove this line if condition change - removing docs ranked first
    df = df[df.previous_pos != 1]

    df.to_csv(f"feature_data_{cp}_new.csv", index=False)
    print("Part 2 ended")
    exit()

else:
    df = pd.read_csv(f"feature_data_{cp}_new.csv")