import pandas as pd
import re


def parse_trec_text_to_dataframe(trec_text):
    # Regular expression patterns for extracting DOCNO and TEXT content
    docno_pattern = r"<DOCNO>(.*?)<\/DOCNO>"
    text_pattern = r"<TEXT>(.*?)<\/TEXT>"

    # Find all matches for DOCNO and TEXT in the input TRECTEXT
    doc_ids = re.findall(docno_pattern, trec_text)
    doc_texts = re.findall(text_pattern, trec_text, flags=re.DOTALL)

    # Create a Pandas DataFrame with the extracted data
    df = pd.DataFrame({'DOCNO': doc_ids, 'TEXT': doc_texts})

    return df


def read_trectext_file(file_path):
    with open(file_path, 'r') as file:
        trec_text = file.read()

    return trec_text


def convert_trectext_to_dataframe(file_path):
    trec_text = read_trectext_file(file_path)
    df = parse_trec_text_to_dataframe(trec_text)
    return df

def read_positions_file(file_path):
    # Read the data from the file with a space delimiter and specify column names
    # column_names = ["query_id", "unknown_col", "docno", "position"]
    column_names = ["id", "Q0", "docno", "0", "score", "name"]
    df = pd.read_csv(file_path, delimiter=' ', names=column_names)
    df["query_id"] = df.docno.str.split("-", expand=True)[2]
    df["position"] = df.groupby('id')['score'].rank(method='first', ascending=False).astype(int)
    df = df[["query_id", "docno", "position"]]

    return df


file_path = "/lv_local/home/niv.b/CharPDM/data_greg/documents.trectext"
df = convert_trectext_to_dataframe(file_path)
df[["round_no", "query_id", "username"]] = df["DOCNO"].str.split("-", expand=True).drop(0, axis=1)
queries = {
    "002": "french lick resort casino",
    "004": "toilet",
    "009": "car parts",
    "010": "cheap internet",
    "011": "gmat prep classes",
    "017": "poker tournaments",
    "018": "wedding budget calculator",
    "029": "ps 2 games",
    "032": "website design hosting",
    "033": "elliptical trainer",
    "034": "cell phones",
    "036": "gps",
    "045": "solar panels",
    "048": "wilson antenna",
    "051": "horse hooves",
    "059": "build fence",
    "069": "sewing instructions",
    "078": "dieting",
    "098": "sat",
    "124": "bowflex power pro",
    "144": "trombone sale",
    "161": "furniture small spaces",
    "164": "hobby stores",
    "166": "computer programming",
    "167": "barbados",
    "177": "best long term care insurance",
    "180": "newyork hotels",
    "182": "quit smoking",
    "188": "internet phone service",
    "193": "dog clean bags",
    "195": "pressure washers"}

rows = [{"query_id": k, "query": v} for k, v in queries.items()]
query_df = pd.DataFrame(data=rows)
df = df.merge(how="left", on="query_id", right=query_df).rename({"TEXT": "current_document", "DOCNO": "docno"}, axis=1)
df["group"] = "A"
# current_document,docno,group,position,query,query_id,round_no,username
file_path = "/lv_local/home/niv.b/CharPDM/data_greg/trec_file_original_sorted.txt"  # Replace this with the actual file path
pos_df = read_positions_file(file_path)
df = df.merge(pos_df[["docno","position"]], how="outer", on="docno")
df = df[df.position.notna()]
df.position = df.position.astype(int)
df.to_csv("greg_data.csv", index=False)
print(df)
x=1
