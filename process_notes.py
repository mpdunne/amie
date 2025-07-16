import os
import glob
import pickle
import pandas as pd

from openai import OpenAI
from dotenv import load_dotenv

# Config
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RAW_DATA_DIR = "raw"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
VOCAB_OUTPUT = os.path.join(DATA_DIR, 'vocab.csv')
processed_suffix = '.processed.pkl'


# Vocab extractor
def extract_vocab_from_notes(notes_text):
    prompt = f"""
Here are my messy French notes from lessons. Note that there might not actually be any French in here! If it looks like there's no vocab, just return an empty string. Sometimes there isn't—some files aren't relevant.

But if there is, please extract each French word/concept/phrase.  
For each one, return JUST the word/concept/phrase—but if there's a specific or unusual English usage (for example an ambiguity), add that in brackets after the word/phrase.

Return a \\n-separated string. Each word/phrase on its own line. NO EXTRA TEXT. I'll process it later.

Here are my notes:
{notes_text}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful French language assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


# Read and process all notes
all_vocab = []

for file in glob.glob(os.path.join(RAW_DATA_DIR, '**/*'), recursive=True):
    if file.endswith(('.txt', '.md', '.csv')):
        processed_file = f'{file}{processed_suffix}'

        if not os.path.exists(processed_file):
            print(f'Reading {file}')
            content = open(file, 'r').read()
            content_processed = extract_vocab_from_notes(content)
            print(content_processed)
            pickle.dump(content_processed, open(processed_file, 'wb'))
        else:
            print(f'Loading {file}...')
            content_processed = pickle.load(open(processed_file, 'rb'))

        vocab = [x.strip() for x in content_processed.split('\n')]
        all_vocab += vocab


# Merge with existing vocab CSV
def merge_vocab_list_to_csv(vocab_list, csv_file):
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=['Entry', 'Notes', 'Mastery Score'])

    vocab_set = set(filter(None, vocab_list))
    existing_entries = set(df['Entry'].astype(str))
    new_words = vocab_set - existing_entries

    new_rows = pd.DataFrame({
        'Entry': list(new_words),
        'Notes': '',
        'Mastery Score': 0
    })

    combined_df = pd.concat([df, new_rows], ignore_index=True).drop_duplicates(subset=['Entry'])
    combined_df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"Added {len(new_words)} new words. Total vocab size: {len(combined_df)}.")


merge_vocab_list_to_csv(all_vocab, VOCAB_OUTPUT)
