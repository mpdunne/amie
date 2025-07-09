import os
import re
import pandas as pd

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
DATA_DIR = os.path.join(BASE_DIR, 'data')
VOCAB_CSV = os.path.join(DATA_DIR, 'vocab.csv')


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

def amie_says(text):
    print(f"\n{Fore.MAGENTA}Amie : {text}{Style.RESET_ALL}")


def run_vocab_chat():
    amie_says("Bienvenue en mode vocabulaire !")

    df = load_vocab(VOCAB_CSV)

    low_mastery = df[df['Mastery Score'] <= 7]
    session_words = low_mastery.sample(n=min(10, len(low_mastery)))['Entry'].tolist()
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

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        reply = response.choices[0].message.content
        amie_says(reply)

        user_input = input("\nToi : ").strip()

        if user_input.startswith('/'):
            # Save progress before leaving
            rating_prompt = f"""
Based on our conversation, rate my mastery for these words on a scale from 1 (poor) to 10 (excellent).
Give your rating for each word:
{vocab_text}
"""
            messages.append({"role": "user", "content": rating_prompt})

            rating_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )

            amie_says("Voici mon évaluation de ta maîtrise :")
            amie_says(rating_response.choices[0].message.content)

            # Extract ratings from GPT response
            for line in rating_response.choices[0].message.content.split('\n'):
                for word in session_words:
                    if word in line:
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            rating = int(numbers[0])
                            df.loc[df['Entry'] == word, 'Mastery Score'] = rating

            save_vocab(df, VOCAB_CSV)
            amie_says("Vocabulaire enregistré.")
            return user_input  # Escalate command to main loop

        # Detect bracketed words
        bracketed = re.findall(r'\[(.*?)\]', user_input)
        if bracketed:
            amie_says(f"Mots inconnus détectés : {', '.join(bracketed)}")
            for word in bracketed:
                if word not in df['Entry'].values:
                    df.loc[len(df)] = {'Entry': word, 'Notes': '', 'Mastery Score': 0}

        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": user_input})


def run_general_chat():
    # Amie starts the conversation herself
    messages = [
        {"role": "system", "content": "Tu es une amie française très sympathique. "
                                     "Discute de manière détendue et amicale. "
                                     "Parle uniquement en français. Pose des questions légères et amusantes si possible."},
        {"role": "user", "content": "Commence la conversation en parlant de choses amusantes."}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    reply = response.choices[0].message.content
    amie_says(reply)

    while True:
        user_input = input("\nToi : ").strip()

        if user_input.startswith('/'):
            return user_input

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        reply = response.choices[0].message.content
        amie_says(reply)
        messages.append({"role": "assistant", "content": reply})


def run_chat():
    amie_says("Salut ! C'est moi, Amie! Alors....")

    current_mode = 'chat'  # Start in general chat

    while True:

        if current_mode == 'chat':
            returned_command = run_general_chat()

        elif current_mode == 'vocab':
            returned_command = run_vocab_chat()

        else:
            amie_says("Mode inconnu.")
            run_general_chat()

        # Handle slash command returned from mode
        if returned_command:
            command = returned_command.lower()

            if command in ('/exit', '/quit', '/done', '/qq'):
                amie_says("À bientôt !")
                break

            elif command in ('/vocab', '/vocabulaire', '/v'):
                current_mode = 'vocab'
                amie_says("Passage en mode vocabulaire.")

            elif command in ('/chat', '/discussion', '/c'):
                current_mode = 'chat'
                amie_says("Retour en mode chat.")

            else:
                amie_says("Commande inconnue.")
                # Stay in current mode


###########
# Run it! #
###########

if __name__ == '__main__':
    run_chat()
