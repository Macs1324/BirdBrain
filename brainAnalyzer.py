import quickbrain as qb
import QuickMaths as qm

#A function that checks the weights and biases
#idk why past me decided to put this in a separate file

def print_brain(brain):
    for i in brain.weights:
        print("WEIGHT:")
        print(i)
    for i in brain.biases:
        print("BIAS")
        print(i)
