import random
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from train import all_words, tags, intents

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

FILE = "model/FFN.pth"
model = torch.load(FILE)

def chatbot(message):
    model.eval()
    while True:
        # sentence = input('You : ')
        sentence = message
        print(f"Human : {sentence}")
        if sentence == "quit":
            break

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    response = random.choice(intent["responses"])
                    print(f'Bot : {response}')
                    return response
        else:
            response = "I'm sorry i dont understand"
            print(f"Bot : {response}")
            return response

# chatbot()


def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Hello! Welcome.")
                         

def handle_message(update, context):
    message_text = update.message.text
    response = chatbot(message_text)
    # context.bot.send_message(chat_id=update.effective_chat.id, text=f"You said: {message_text}")
    context.bot.send_message(chat_id=update.effective_chat.id, text=f"{response}")

# Set up the bot
updater = Updater(token='6035117137:AAFBsUhD38QM9Jb_O4Run-KEKcYENpWtx2I', use_context=True)

# Get the dispatcher to register handlers
dispatcher = updater.dispatcher

# Add handlers for commands and messages
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

# Start the bot
updater.start_polling()
