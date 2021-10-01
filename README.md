!pip install hmmlearn==0.2.2

#definisikan  Matriks Transisi (states)
states = ('Rainy', 'Sunny')
print(states)

#definisi Matriks Observasi / Matriks Emisi
observations = ('walks', 'shop', 'clean')
print(observations)

#defisi Matrik Priority
start_probability = {'Rainy': 0.6, 'Sunny': 0.4}
print(start_probability)

transition_probability = {
    'Rainy' : {'Rainy': 0.7, 'Sunny': 0.4},
    'Sunny' : {'Rainy': 0.6, 'Sunny': 0.3}
}
print(transition_probability)

emission_probability = {
    'Rainy' : {'walk': 0.1, 'shop': 0.5, 'clean': 0.4},
    'Sunny' : {'walk': 0.6, 'shop': 0.3, 'clean': 0.1}
}
print(emission_probability)

# Library untuk HMM
# Cara install pip install hmmlearn==0.2.2

from hmmlearn import hmm
import numpy as np
  
model = hmm.MultinomialHMM(n_components=2)
model.startprob_ = np.array([0.6, 0.4])
model.transmat_ = np.array([[0.7, 0.3],
                            [0.4, 0.6]])
model.emissionprob_ = np.array([[0.1, 0.4, 0.5],
                                [0.6, 0.3, 0.1]])
                                
