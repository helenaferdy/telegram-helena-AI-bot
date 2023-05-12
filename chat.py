import random
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from train import all_words, tags, intents

FILE = "model/FFN.pth"
model = torch.load(FILE)

def chatbot():
    model.eval()
    while True:
        sentence = input('You : ')
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