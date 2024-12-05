# -----------------------------------------------------------------------------
# Copyright 2024 (c) Alexis Daugé
# Released under a BSD two-clauses license
#
# References: Eugene Charniak (2018). Introduction to Deep Learning. MIT Press.
# -----------------------------------------------------------------------------
from math import exp

def softmax(z: list[int], i: int):
    """Compute softmax values for each sets of scores in vector z."""
    return exp(z[i]) / sum(exp(z[j]) for j in range(len(z)))

if __name__ == "__main__":
    z = [-1, 0, 1]
    print([softmax(z, i) for i in range(len(z))]) # [0.090, 0.244, 0.665]
