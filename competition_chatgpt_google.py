import openai
import config
import tiktoken
import pandas as pd
import re
import warnings
from nltk.corpus import stopwords
from config import current_prompt as cp
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

warnings.filterwarnings("ignore")

encoder = tiktoken.encoding_for_model(config.model)

stop_words = set(stopwords.words('english'))


def flatten_list(data):
    result = []

    for item in data:
        if isinstance(item, dict):
            result.append(item)
        elif isinstance(item, list):
            result.extend(flatten_list(item))

    return result


def get_top_user(data, r):
    df = data[data.round_no == r][["username", "position"]].set_index("username")

    return df.median(axis=1).idxmin()


def remove_sentences_second(sentences):
    # Calculate the initial total number of words
    total_words = sum(len(sentence.split()) for sentence in sentences)

    # Check if total_words is already within the desired range or below 120
    if total_words <= 150:
        return sentences

    # Find and remove a sentence if total_words can be adjusted within the desired range
    while total_words > 150:
        for sentence in sorted(sentences, key=len):
            # Calculate the updated total number of words without the current sentence
            updated_total_words = total_words - len(sentence.split())
            if updated_total_words <= 150:
                total_words = updated_total_words
                sentences.remove(sentence)
                return sentences

        # Remove the shortest sentence if no sentence can be removed to meet the desired range or below 120
        if total_words > 150:
            shortest_sentence = min(sentences, key=lambda sentence: len(sentence))
            sentences.remove(shortest_sentence)
            total_words = total_words - len(shortest_sentence.split())
    return sentences


def count_words_complete_sentences(text):
    # Split the text into sentences using a regex pattern
    sentences = re.split(r'(?<=[.!?])\s+|\n', text)

    # Check if the last sentence is incomplete and remove it if necessary
    if sentences and sentences[-1].strip() and sentences[-1].strip()[-1] not in ['.', '!', '?']:
        sentences.pop()

    sentences_second = sentences.copy()

    # Check if truncation is necessary
    if sentences:
        word_count = sum(len(sentence.split()) for sentence in sentences)
        truncated_text = " ".join(sentences)

        while word_count > 150:
            if len(sentences) < 2:
                break
            sentences.pop()
            truncated_text = ' '.join(sentences)
            word_count = sum(len(sentence.split()) for sentence in sentences)

        if word_count < 120:
            # second try (less preferred)
            word_count_second = sum(len(sentence.split()) for sentence in sentences_second)
            sentences_second = remove_sentences_second(sentences_second)
            truncated_text_second = ' '.join(sentences_second)
            word_count_second_new = sum(len(sentence.split()) for sentence in sentences_second)
            print(f"second try initiated, word counts - orig: {word_count_second}, new: {word_count_second_new}")

            if word_count_second_new >= 120 and word_count_second_new <= 150:
                return word_count_second_new, truncated_text_second + "." if truncated_text_second[
                                                                                 -1] != "." else truncated_text_second, True
            if word_count < 120:
                return len(text.split()), text, False

        if word_count >= 120 and word_count <= 150:
            return word_count, truncated_text + "." if truncated_text[-1] != "." else truncated_text, True

    # All sentences are complete
    word_count = len(text.split())
    return word_count, text, False


def get_messages(bot_name, creator_name, data, query_id):
    assert data is not None
    messages = config.get_prompt(bot_name, data, creator_name, query_id)
    tokens = sum(len(encoder.encode(p['content'])) for p in messages if 'content' in p)
    assert tokens <= 3500, f"Prompt too long, too many tokens: {tokens}"

    return flatten_list(messages)


def get_comp_text(messages, temperature=config.temperature, top_p=config.top_p,
                  frequency_penalty=config.frequency_penalty, presence_penalty=config.presence_penalty):
    max_tokens = config.max_tokens
    response = False
    # prompt_tokens = len(encoder.encode("".join([line['content'] for line in messages]))) + 200
    # while prompt_tokens + max_tokens > 4096:
    #     max_tokens -= 50
    #     print("Changed max tokens for response to:", max_tokens)

    word_no, res, counter = 0, "", 0

    while not response:
        try:
            response = openai.ChatCompletion.create(
                model=config.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )

            # word_no, res, ok_flag = count_words_complete_sentences(response['choices'][0]['message']['content'])
            res = response['choices'][0]['message']['content']
            word_no = len(res.split())
            # if counter > 5:
            #     print("LOOP BREAK - Try creating a new text manually. Truncated.")
            #     res = " ".join(res.split()[:148]) + "."
            #     counter = 0
            #     break
            # if word_no < 120 or word_no > 150:
            #     max_tokens += 10
            #     response = False
            #     print(f"word no was: {word_no}, increasing max tokens to: {max_tokens}.")
            #     counter += 1
            #     continue
            # counter = 0
            break
        except Exception as e:
            print(e)
            continue
    print(f"word no is: {word_no}, current max tokes: {max_tokens}.")
    return res


lock = Lock()


def parallel_function(idx, row, data, orig, len_):
    try:
        data = data[data["round_no"] < row["round_no"]]
        bot_name = row["username"]
        creator_name = row["creator"]
        query_id = row["query_id"]
        temp = row["temp"]
        print(
            f"Starting {idx + 1}/{len_}: bot: {bot_name}, creator: {creator_name}, query:{query_id}, round: {row['round_no']}")
        # rel_data = data[data['query_id'] == query_id]
        messages = get_messages(bot_name, creator_name, data, query_id)
        res = get_comp_text(messages,temperature=temp)
        with lock:
            orig.at[idx, "prompt"] = str(messages)
            orig.at[idx, "text"] = res
            orig.to_csv(f"bot_followup_{cp}.csv", index=False)
        print(
            f"Done {idx + 1}/{len_} ({len_ - idx - 1} left): bot: {bot_name}, creator: {creator_name}, query:{query_id}, round: {row['round_no']}")
    except Exception as e:
        print(f"An error occurred for index {idx}: {e}")


if __name__ == '__main__':
    orig = pd.read_csv(f"bot_followup_{cp}.csv")
    bot_followup = orig[orig['text'].isna()]
    data = pd.read_csv('greg_data.csv')

    # if 'STATBOT' not in list(orig.username.unique()):
    #     stat_df = pd.merge(orig[['round_no', 'query_id', 'creator']].drop_duplicates(), data, how='left',
    #                        right_on=['round_no', 'query_id', 'username'],
    #                        left_on=['round_no', 'query_id', 'creator']).rename({"current_document": "text"}, axis=1)[
    #         bot_followup.columns]
    #     stat_df["username"] = "STATBOT"
    #     stat_df['text'] = stat_df['text'].str.strip()
    #     orig = pd.concat([orig, stat_df], ignore_index=True)
    #     orig.to_csv(f"bot_followup_{cp}.csv", index=False)

    len_ = len(orig)

    # debugging:
    # for idx, row in bot_followup.iterrows():
    #     parallel_function(idx, row, data, orig, len_)

    with ThreadPoolExecutor(max_workers=1) as executor:  # Change max_workers as needed
        futures = {executor.submit(parallel_function, idx, row, data.copy(), orig, len_): row for idx, row in
                   bot_followup.iterrows()}
        for future in as_completed(futures):
            row = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred in future: {e}")

    print("All tasks completed.")
