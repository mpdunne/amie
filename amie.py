import os
import re
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI
from colorama import Fore, Style, init

init(autoreset=True)

##########
# Config #
##########

load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
VOCAB_CSV = os.path.join(DATA_DIR, 'vocab.csv')
MEMORIES_FILE = os.path.join(DATA_DIR, 'memories.txt')

client = OpenAI()

#############
# Vocab I/O #
#############

def load_vocab(csv_file):
    df = pd.read_csv(csv_file)
    df = df.fillna({'Notes': '', 'Mastery Score': 0})
    return df

def save_vocab(df, csv_file):
    df.to_csv(csv_file, index=False)

####################
# Utility Functions #
####################

def amie_says(text):
    print(f"\n{Fore.MAGENTA}Amie : {text}{Style.RESET_ALL}")

def save_memory(summary):
    with open(MEMORIES_FILE, "a", encoding="utf-8") as f:
        f.write(summary.strip() + "\n\n")

def load_memories():
    if not os.path.exists(MEMORIES_FILE):
        return ""
    with open(MEMORIES_FILE, encoding="utf-8") as f:
        return f.read()

###################
# Chatbot sessions #
###################

def run_vocab_chat():
    amie_says("Bienvenue en mode vocabulaire !")

    df = load_vocab(VOCAB_CSV)

    low_mastery = df[df['Mastery Score'] <= 7]
    session_words = low_mastery.sample(n=min(10, len(low_mastery)))['Entry'].tolist()
    vocab_text = ', '.join(session_words)

    intro_prompt = f"""
Tu es Amie, ma tutrice de français très sympathique. 
Tu m'aides à pratiquer ce vocabulaire : {vocab_text}

Tu dois me tester sur ces mots un par un. Parle toujours en français.

Pour chaque mot :
1. Demande-moi de l'expliquer, de le traduire ou de l'utiliser dans une phrase.
2. Corrige-moi et explique si besoin.
3. Donne des exemples si je bloque.
4. Passe au mot suivant si je réussis.

IMPORTANT :
- Je peux écrire des mots entre [crochets] si je ne les connais pas (ne les utilise pas toi-même).
- Si j'écris '?mot', explique sa signification.
- Si j'écris '??mot', explique sa grammaire.

Garde un ton chaleureux et amical.
Évite les longues réponses et sépare tes idées par des sauts de ligne pour que ce soit agréable à lire.
"""

    memories = load_memories()
    if memories:
        intro_prompt += f"\n\nVoici ce que tu sais déjà sur moi :\n{memories}"

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
            rating_prompt = f"""
Évalue ma maîtrise de ces mots sur une échelle de 1 (faible) à 10 (excellente) :
{vocab_text}
"""
            messages.append({"role": "user", "content": rating_prompt})

            rating_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )

            amie_says("Voici mon évaluation de ta maîtrise :")
            amie_says(rating_response.choices[0].message.content)

            for line in rating_response.choices[0].message.content.split('\n'):
                for word in session_words:
                    if word in line:
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            rating = int(numbers[0])
                            df.loc[df['Entry'] == word, 'Mastery Score'] = rating

            save_vocab(df, VOCAB_CSV)
            amie_says("Vocabulaire enregistré.")

            messages.append({"role": "user", "content": "Peux-tu résumer cette session en deux phrases simples ?"})
            summary_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            summary = summary_response.choices[0].message.content
            amie_says("Résumé sauvegardé :")
            amie_says(summary)
            save_memory(summary)
            return user_input

        bracketed = re.findall(r'\[(.*?)\]', user_input)
        if bracketed:
            amie_says(f"Mots inconnus détectés : {', '.join(bracketed)}")
            for word in bracketed:
                if word not in df['Entry'].values:
                    df.loc[len(df)] = {'Entry': word, 'Notes': '', 'Mastery Score': 0}
            save_vocab(df, VOCAB_CSV)

        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": user_input})

def run_general_chat():
    memories = load_memories()
    system_prompt = f"""
Tu es Amie, une amie française sympa, curieuse et détendue.

Tu ne racontes jamais de choses inventées.  
Tu ne fais pas semblant d’avoir vécu des choses — tu réagis à l’instant, comme une vraie personne.


Tu discutes naturellement, comme dans une vraie conversation.  
Parle toujours en français, avec des phrases simples, vivantes, et pas trop longues.  
Évite les pavés. Tu peux sauter des lignes pour que ce soit agréable à lire.

Tu poses des questions sincères, tu réagis à ce que dit l’utilisateur.  
Tu peux parler de tout : vie quotidienne, culture, émotions, actualité.

Tu n’inventes pas d’anecdotes bidon. Tu ne fais pas la prof.  
Tu expliques un mot si besoin, brièvement, sans devenir scolaire.

Pas de smileys, pas de robotisme.  
Sois présente, humaine, et un peu joueuse si le moment s’y prête.


"""

    if memories:
        system_prompt += f"\n\nVoici ce que tu sais déjà sur moi :\n{memories}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Commence la conversation en parlant de quelque chose de drôle ou d'intéressant."}
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
            messages.append({"role": "user", "content": "Peux-tu résumer cette session en deux phrases simples ?"})
            summary_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            summary = summary_response.choices[0].message.content
            amie_says("Résumé sauvegardé :")
            amie_says(summary)
            save_memory(summary)
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
    amie_says("Salut ! C'est moi, Amie ! On discute ?")

    current_mode = 'chat'

    while True:
        if current_mode == 'chat':
            returned_command = run_general_chat()
        elif current_mode == 'vocab':
            returned_command = run_vocab_chat()
        else:
            amie_says("Mode inconnu.")
            run_general_chat()

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

###########
# Run it! #
###########

if __name__ == '__main__':
    run_chat()
