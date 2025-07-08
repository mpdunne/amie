import pandas as pd
import os
import re

from dotenv import load_dotenv

from openai import OpenAI
client = OpenAI()

from colorama import Fore, Style, init
init(autoreset=True)

##########
# Config #
##########

# OpenAI stuff
load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

# Directory of the script itself
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths relative to the script
VOCAB_CSV = os.path.join(BASE_DIR, 'vocab.csv')

#############
# Vocab I/O #
#############

def load_vocab(csv_file):
    df = pd.read_csv(csv_file)
    df = df.fillna({'Notes': '', 'Mastery Score': 0})
    return df

def save_vocab(df, csv_file):
    df.to_csv(csv_file, index=False)

###################
# Chatbot session #
###################
    
def run_vocab_chat(df, num_words=10):
    # Select low-mastery words
    low_mastery = df[df['Mastery Score'] <= 7]
    session_words = low_mastery.sample(n=min(num_words, len(low_mastery)))['Entry'].tolist()

    #print("Words for this session:", session_words)

    # Initial GPT prompt
    vocab_text = ', '.join(session_words)
    intro_prompt = f"""
You are my French tutor. Your name is Amie.

You have a list of words for me to practice: {vocab_text}

Your task is to test me on these words, one by one. You should speak to me entirely in French.

For each word:
1. Ask me to explain or translate it, or to use it in a French sentence.
2. Check my answer and explain if necessary.
3. Provide correct examples and additional explanations if I struggle.

If I answer correctly, move on to the next word.

IMPORTANT:
- I might write words inside [square brackets] to flag words I don’t know. This is just for me—you should NEVER use square brackets yourself.
- If I type `?word`, explain the meaning or usage of that word.
- If I type `??word`, explain grammar, conjugation, or nuances of that word.

Focus purely on helping me practice these words effectively. Keep your questions short and clear.
"""
    messages = [
        {"role": "system", "content": "You are a helpful French tutor."},
        {"role": "user", "content": intro_prompt}
    ]

    # Start conversation loop
    while True:
        response = client.chat.completions.create(model="gpt-4o",
        messages=messages)
        reply = response.choices[0].message.content
        print(f"\n{Fore.MAGENTA}Amie: {reply}{Style.RESET_ALL}")

        user_input = input("\nYou: ")
        if user_input.lower().strip() in ('/exit', '/quit', '/save'):
            print("Ending session.")
            break

        # Detect bracketed words
        bracketed = re.findall(r'\[(.*?)\]', user_input)
        if bracketed:
            print("Detected unknown words:", bracketed)
            for word in bracketed:
                if word not in df['Entry'].values:
                    df.loc[len(df)] = {'Entry': word, 'Notes': '', 'Mastery Score': 0}

        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": user_input})

    # Ask GPT to rate mastery
    rating_prompt = f"""
Based on our conversation, rate my mastery for these words on a scale from 1 (poor) to 10 (excellent).
Give your rating for each word:
{vocab_text}
"""
    messages.append({"role": "user", "content": rating_prompt})

    rating_response = client.chat.completions.create(model="gpt-4o",
    messages=messages)

    print("\nAmie's Assessment:")
    print(rating_response.choices[0].message.content)

    # Extract ratings from GPT response
    for line in rating_response.choices[0].message.content.split('\n'):
        for word in session_words:
            if word in line:
                numbers = re.findall(r'\d+', line)
                if numbers:
                    rating = int(numbers[0])
                    df.loc[df['Entry'] == word, 'Mastery Score'] = rating

    save_vocab(df, VOCAB_CSV)
    print("\nVocab saved.")

###########
# Run it! #
###########

if __name__ == '__main__':
    df = load_vocab(VOCAB_CSV)
    run_vocab_chat(df)

