'''
hidden_states = ('Rainy', 'Sunny', 'Cloudy')

obs_states = ('sleep', 'game', 'eat', )

start_prob = {'Rainy': 0.3, 'Sunny': 0.4, 'Cloudy': 0.3}

trans_prob = {'Rainy'  : {'Rainy': 0.4, 'Sunny': 0.3, 'Cloudy': 0.3},
              'Sunny'  : {'Rainy': 0.2, 'Sunny': 0.7, 'Cloudy': 0.1},
              'Cloudy' : {'Rainy': 0.4, 'Sunny': 0.1, 'Cloudy': 0.5}
             }

em_prob = {'Rainy'  : {'sleep': 0.4, 'game': 0.4, 'eat': 0.1},
           'Sunny'  : {'sleep': 0.2, 'game': 0.7, 'eat': 0.1},
           'Cloudy' : {'sleep': 0.2, 'game': 0.2, 'eat': 0.6},
          }
'''


obs_states = ('normal', 'cold', 'dizzy', )

hidden_states = ('Healthy', 'Fever')

start_prob = {'Healthy': 0.6, 'Fever': 0.4}

trans_prob = {'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
              'Fever'   : {'Healthy': 0.4, 'Fever': 0.6}
             }

em_prob = {'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
           'Fever'   : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
          }


def forward_viterbi(obs_states, hidden_states, start_prob, trans_prob, em_prob):

    T = {}
    # T(t = 0)
    for s in hidden_states:

        p = em_prob[s][obs_states[0]] * start_prob[s]
        T[s] = p, [s], p # total_prob, viterbi_path, viterbi_prob

    # T(t > 0)
    for t in range(1, len(obs_states)):
        U = {}
        for s0 in hidden_states:

            total_prob, max_path, max_prob = 0, None, 0
            for s1 in hidden_states:

                p = em_prob[s1][obs_states[t]] * trans_prob[s0][s1]

                prob, viterbi_path, viterbi_prob = T[s0]

                prob *= p
                total_prob += prob

                viterbi_prob *= p
                if viterbi_prob > max_prob:
                    max_path = viterbi_path + [s1]
                    max_prob = viterbi_prob

            U[s0] = total_prob, max_path, max_prob

        T = U

    # find sum/max to the final states:
    total_prob, max_path, max_prob = 0, None, 0
    for s in hidden_states:
        #print(T[s])
        prob, viterbi_path, viterbi_prob = T[s]
        total_prob += prob

        if viterbi_prob > max_prob:
            max_path = viterbi_path
            max_prob = viterbi_prob

    return total_prob, max_path, max_prob


if __name__ == '__main__':

    total_prob, max_path, max_prob = forward_viterbi(obs_states, hidden_states, start_prob, trans_prob, em_prob)

    print('Results')
    print(total_prob)
    print(max_path)
    print(max_prob)
