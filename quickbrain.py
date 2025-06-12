import json
import os
import pickle
import random
import statistics
import time

import QuickMaths as qm

# from PIL import Image
# import matplotlib.pyplot as plt


# Remember the 15 year old me that made QuickMaths? He made this too
# Warning: A tooon of broken bloat classes


class Basic_3L:
    def __init__(self, n_i, n_h, n_o):
        self.i = qm.Matrix(n_i, 1)

        self.wih = qm.Matrix(n_h, n_i)

        self.h = qm.Matrix(n_h, 1)

        self.who = qm.Matrix(n_o, n_h)

        self.o = qm.Matrix(n_o, 1)

        self.i.clear()
        self.wih.randomize()
        self.h.clear()
        self.who.randomize()
        self.o.clear()

        """
    def mutate(self, rate, seed=None):
		if seed != None:
			qm.np.random.seed(seed)
		mutation = qm.Matrix(self.n_h, self.n_i)
		mutation.randomize()ù
		"""

    def feed_forward(self, q, activation=qm.linear):
        self.i = q
        self.h = self.wih.times(self.i)
        self.h.applyFunc(activation)
        self.o = self.who.times(self.h)
        self.o.applyFunc(activation)

        return self.o

    def printValues(self):
        self.i.printValue("Input layer")
        self.wih.printValue("First weights")
        self.h.printValue("Hidden layer")
        self.who.printValue("Second weights")
        self.o.printValue("Output layer")


class Genetic_3L:
    def __init__(self, n_i, n_h, n_o, n_bots):
        self.n_i = n_i
        self.n_o = n_o
        self.n_bots = n_bots
        self.bots = [Basic_3L(n_i, n_h, n_o) for i in range(n_bots)]
        for bot in self.bots:
            bot.randomize_weights()

    def get_outputs(self, q):
        self.bot_outputs = []
        for bot in self.bots:
            self.bot_outputs.append(bot.feed_forward(q, qm.sigmoid))

    def calculate_costs(self, a):
        self.bot_costs = []
        self.bot_errors = []
        for out in self.bot_outputs:
            self.bot_errors.append(a.minus(out))
        for err in self.bot_errors:
            self.bot_costs.append(qm.cost(err))

    def get_best_bot(self):
        best_bot_index = self.bot_costs.index(min(self.bot_costs))
        self.best_bot = self.bots[best_bot_index]

    def get_new_gen(self):
        self.bots = [self.best_bot for i in range(self.n_bots)]

    def mutate_babies(self):
        for bot in self.bots[1:]:
            qm.np.random.seed(self.bots.index(bot))
            bot.randomize_weights()

    def neat(self, q_set, a_set, generations):
        for epoch in range(generations):
            qm.np.random.seed(epoch)
            print("training generation: " + str(epoch))
            rand_choice = qm.random.randint(0, q_set.rows - 1)
            curr_q = qm.Matrix(1, self.n_i)
            curr_a = qm.Matrix(1, self.n_o)
            curr_q.assign([q_set.row(rand_choice)])
            curr_a.assign([a_set.row(rand_choice)])

            curr_q.transpose("replace")
            curr_a.transpose("replace")

            self.get_outputs(curr_q)
            self.calculate_costs(curr_a)
            self.get_best_bot()
            self.get_new_gen()
            self.mutate_babies()
            print(self.bot_costs)
            self.best_bot.feed_forward(curr_q, qm.sigmoid).printValue(
                "guess of the best bot:"
            )
        return self.best_bot


class DFF_5L:
    def __init__(self, n_i, n_h1, n_h2, n_h3, n_o):
        self.n_i = n_i
        self.n_h1 = n_h1
        self.n_h2 = n_h2
        self.n_h3 = n_h3
        self.n_o = n_o
        self.i = qm.Matrix(n_i, 1)

        self.wih1 = qm.Matrix(n_h1, n_i)

        self.h1 = qm.Matrix(n_h1, 1)
        self.b_h1 = qm.Matrix(n_h1, 1)

        self.wh1h2 = qm.Matrix(n_h2, n_h1)

        self.h2 = qm.Matrix(n_h2, 1)
        self.b_h2 = qm.Matrix(n_h2, 1)

        self.wh2h3 = qm.Matrix(n_h3, n_h2)

        self.h3 = qm.Matrix(n_h3, 1)
        self.b_h3 = qm.Matrix(n_h3, 1)

        self.wh3o = qm.Matrix(n_o, n_h3)

        self.o = qm.Matrix(n_o, 1)
        self.b_o = qm.Matrix(n_o, 1)

        self.wih1.randomize()
        self.wh1h2.randomize()
        self.wh2h3.randomize()
        self.wh3o.randomize()

        self.b_h1.clear()
        self.b_h2.clear()
        self.b_h3.clear()
        self.b_o.clear()

    def feed_forward(self, q, activation=qm.sigmoid):
        self.i = q

        self.h1 = self.wih1.times(self.i)
        self.h1.addMatrix(self.b_h1)
        self.h1.applyFunc(activation)

        self.h2 = self.wh1h2.times(self.h1)
        self.h2.addMatrix(self.b_h2)
        self.h2.applyFunc(activation)

        self.h3 = self.wh2h3.times(self.h2)
        self.h3.addMatrix(self.b_h3)
        self.h3.applyFunc(activation)

        self.o = self.wh3o.times(self.h3)
        self.o.addMatrix(self.b_o)
        self.o.applyFunc(activation)

        return self.o

    def get_genotype(self):
        # self.wih1.printValue()
        vector1 = self.wih1.matrix
        vector2 = self.wh1h2.matrix
        vector3 = self.wh2h3.matrix
        vector4 = self.wh3o.matrix

        process = []

        process.append(vector1)
        process.append(vector2)
        process.append(vector3)
        process.append(vector4)

        return process

    def set_genotype(self, gen):
        self.wih1.assign(gen[0])
        self.wh1h2.assign(gen[1])
        self.wh2h3.assign(gen[2])
        self.wh3o.assign(gen[3])

    def mutate(self, rate=0.01):
        for i in range(len(self.wih1.matrix)):
            for j in range(len(self.wih1.matrix[i])):
                self.wih1.matrix[i][j] += rate * qm.np.random.randint(-1, 1)

        for i in range(len(self.wh1h2.matrix)):
            for j in range(len(self.wh1h2.matrix[i])):
                self.wh1h2.matrix[i][j] += rate * qm.np.random.randint(-1, 1)
        for i in range(len(self.wh2h3.matrix)):
            for j in range(len(self.wh2h3.matrix[i])):
                self.wh2h3.matrix[i][j] += rate * qm.np.random.randint(-1, 1)
        for i in range(len(self.wh3o.matrix)):
            for j in range(len(self.wh3o.matrix[i])):
                self.wh3o.matrix[i][j] += rate * qm.np.random.randint(-1, 1)

    def train(self, qset, aset, epochs, learning_rate):

        # ----------------------------------------------		INITIALIZATION

        for epoch in range(epochs):
            print(int((epoch / epochs) * 100), "%")
            rand_choice = qm.np.random.randint(0, qset.rows)
            curr_q = qm.Matrix()
            curr_q.assign([qset.row(rand_choice)])

            curr_a = qm.Matrix()
            curr_a.assign([aset.row(rand_choice)])

            curr_a.transpose("replace")
            curr_q.transpose("replace")

            # curr_q.printValue("Question")
            # curr_a.printValue("Answer")

            # -----------------------------------	ERRORS

            guess = self.feed_forward(curr_q, qm.sigmoid)
            # guess.printValue("guess")
            error = curr_a.minus(guess)

            error_h3 = self.wh3o.transpose().times(error)

            error_h2 = self.wh2h3.transpose().times(error_h3)

            error_h1 = self.wh1h2.transpose().times(error_h2)

            # print(cost)

            # --------------------------------		GRADIENTS
            gradients_o = self.o.with_applied(qm.dsigmoid)
            gradients_o = gradients_o.dot(error)
            gradients_o.multiply(learning_rate)

            gradients_h3 = self.h3.with_applied(qm.dsigmoid)
            gradients_h3 = gradients_h3.dot(error_h3)
            gradients_h3.multiply(learning_rate)

            gradients_h2 = self.h2.with_applied(qm.dsigmoid)
            gradients_h2 = gradients_h2.dot(error_h2)
            gradients_h2.multiply(learning_rate)

            gradients_h1 = self.h1.with_applied(qm.dsigmoid)
            gradients_h1 = gradients_h1.dot(error_h1)
            gradients_h1.multiply(learning_rate)

            # ----------------------------------		DELTAS
            delta_wh3o = gradients_o.times(self.h3.transpose())
            delta_wh2h3 = gradients_h3.times(self.h2.transpose())
            delta_wh1h2 = gradients_h2.times(self.h1.transpose())
            delta_wih1 = gradients_h1.times(self.i.transpose())

            # --------------------------------  		ADJUSTING WEIGHTS
            self.wh3o.addMatrix(delta_wh3o)
            self.wh2h3.addMatrix(delta_wh2h3)
            self.wh1h2.addMatrix(delta_wh1h2)
            self.wih1.addMatrix(delta_wih1)

            self.b_o.addMatrix(gradients_o)
            self.b_h3.addMatrix(gradients_h3)
            self.b_h2.addMatrix(gradients_h2)
            self.b_h1.addMatrix(gradients_h1)


class DFF_3L:
    def __init__(self, n_i, n_h, n_o):
        self.i = qm.Matrix(n_i, 1)
        self.wih = qm.Matrix(n_h, n_i)
        self.h = qm.Matrix(n_h, 1)
        self.who = qm.Matrix(n_o, n_h)
        self.o = qm.Matrix(n_o, 1)

        self.b_o = qm.Matrix(n_o, 1)
        self.b_h = qm.Matrix(n_h, 1)

        self.i.clear()
        self.h.clear()
        self.o.clear()
        self.b_o.clear()
        self.b_h.clear()

        self.wih.randomize()
        self.who.randomize()

    def feed_forward(self, q):
        self.i = q

        self.h = self.wih.times(self.i)
        self.h.addMatrix(self.b_h)
        self.h.applyFunc(qm.sigmoid)

        self.o = self.who.times(self.h)
        self.o.addMatrix(self.b_o)
        self.o.applyFunc(qm.sigmoid)

        return self.o

    def train(self, qset, aset, epochs, learning_rate):
        # ----------------------------------------------		INITIALIZATION

        for epoch in range(epochs):
            print(int((epoch / epochs) * 100), "%")
            rand_choice = qm.np.random.randint(0, qset.rows)
            curr_q = qm.Matrix()
            curr_q.assign([qset.row(rand_choice)])

            curr_a = qm.Matrix()
            curr_a.assign([aset.row(rand_choice)])

            curr_a.transpose("replace")
            curr_q.transpose("replace")

            # curr_q.printValue("Question")
            # curr_a.printValue("Answer")

            guess = self.feed_forward(curr_q)
            error_o = curr_a.minus(guess)

            error_h = self.who.transpose().times(error_o)

            gradients_o = self.o.with_applied(qm.dsigmoid)
            gradients_o = gradients_o.dot(error_o)
            gradients_o.multiply(learning_rate)

            gradients_h = self.h.with_applied(qm.dsigmoid)
            gradients_h = gradients_h.dot(error_h)
            gradients_h.multiply(learning_rate)
            gradients_h.printValue("gradients")

            delta_who = gradients_o.times(self.h.transpose())
            delta_wih = gradients_h.times(self.i.transpose())

            self.wih.addMatrix(delta_wih)
            self.who.addMatrix(delta_who)

            self.b_h.addMatrix(gradients_h)
            self.b_o.addMatrix(gradients_o)

    def backpropagate(self, error, learning_rate):
        # error_o = qm.filled_matrix(self.o.rows, self.o.cols, error)
        error_o = error

        error_h = self.who.transpose().times(error_o)

        gradients_o = self.o.with_applied(qm.dsigmoid)
        gradients_o = gradients_o * (error_o)
        gradients_o *= learning_rate

        gradients_h = self.h.with_applied(qm.dsigmoid)
        gradients_h = gradients_h.dot(error_h)
        gradients_h.multiply(learning_rate)
        # gradients_h.printValue("gradients")

        delta_who = gradients_o.times(self.h.transpose())
        delta_wih = gradients_h.times(self.i.transpose())

        self.wih.addMatrix(delta_wih)
        self.who.addMatrix(delta_who)

        self.b_h.addMatrix(gradients_h)
        self.b_o.addMatrix(gradients_o)


class Perceptron:
    def __init__(self, n_i, n_o):
        self.i = qm.Matrix(n_i, 1)
        self.i.clear()

        self.w = qm.Matrix(n_o, n_i)
        self.w.randomize()

        self.o = qm.Matrix(n_o, 1)
        self.o.clear()

    def feed_forward(self, q):
        self.i = q
        self.o = self.w.times(self.i)
        return self.o

    def train(self, qset, aset, epochs, learning_rate):
        for epoch in range(epochs):
            rand_choice = qm.np.random.randint(0, qset.rows)
            curr_q = qm.Matrix()
            curr_q.assign([qset.row(rand_choice)])

            curr_a = qm.Matrix()
            curr_a.assign([aset.row(rand_choice)])
            # curr_a.printValue("answer")

            curr_a.transpose("replace")
            curr_q.transpose("replace")

            guess = self.feed_forward(curr_q)
            guess.printValue("guess")
            print(guess.shape(), curr_a.shape())
            error = curr_a.minus(guess)

            cost = qm.mean_sq_err(error)
            print("error: ", cost)
            # print("cost: ", cost)

            delta = self.i.times(error.transpose()).multiply(learning_rate)
            delta.transpose("replace")
            self.w = qm.matrix_sum(self.w, delta)


def crossover(gen1, gen2, prob, mode):
    assert len(gen1) == len(gen2)
    if mode == "matrices":
        for g in gen1:
            if qm.np.random.randint(0, 2) != 0:
                # print("crossing over matrices")
                g = gen2[gen1.index(g)]
        return gen1
    elif mode == "weights":
        for a in range(len(gen1)):
            for i in range(len(gen1[a])):
                for j in range(len(gen1[a][i])):
                    if qm.np.random.randint(0, prob) == prob - 1:
                        gen1[a][i][j] = gen2[a][i][j]
                        # print("Permutating a weight")
        return gen1
    else:
        print("mode not defined")


def neat(
    n_i,
    n_h1,
    n_h2,
    n_h3,
    n_o,
    q_set,
    a_set,
    epochs,
    gen_size,
    cross_mode,
    activation,
    mutation_rate=0.01,
):
    for epoch in range(epochs):
        bots = [DFF_5L(n_i, n_h1, n_h2, n_h3, n_o) for i in range(gen_size)]
        rand_choice = qm.np.random.randint(0, q_set.rows)
        curr_q = qm.Matrix()
        curr_q.assign([q_set.matrix[rand_choice]])
        curr_q.transpose("replace")

        curr_a = qm.Matrix()
        curr_a.assign([a_set.matrix[rand_choice]])
        curr_a.transpose("replace")

        # curr_q.printValue("Question:")
        # curr_a.printValue("answer:")

        answers = []
        costs = []

        for bot in bots:
            answers.append(bot.feed_forward(curr_q, activation))

        for answer in answers:
            costs.append(qm.cost(curr_a.minus(answer)))

        print("minimum cost:", min(costs))
        mama_index = costs.index(min(costs))
        del costs[mama_index]

        papa_index = costs.index(min(costs))
        del costs[papa_index]

        papa = bots[papa_index]
        mama = bots[mama_index]

        papa_genes = papa.get_genotype()
        # print(papa_genes)
        mama_genes = mama.get_genotype()
        # print(mama_genes)

        baby_genes = crossover(papa_genes, mama_genes, 1, cross_mode)
        # print(baby_genes)
        baby = DFF_5L(n_i, n_h1, n_h2, n_h3, n_o)
        baby.set_genotype(baby_genes)

        bots = [baby for i in range(gen_size)]
        # for bot in bots[1:]:
        # 	bot.mutate(mutation_rate)
    bestbot_index = costs.index(min(costs))
    bestbot = bots[bestbot_index]

    return bestbot


class MultilayerPerceptron:
    def __init__(self, n_i, n_h, n_o):
        self.i = qm.Matrix(n_i, 1)
        self.wih = qm.Matrix(n_h, n_i)
        self.h = qm.Matrix(n_h, 1)
        self.who = qm.Matrix(n_o, n_h)
        self.o = qm.Matrix(n_o, 1)

        self.i.clear()
        self.wih.randomize()
        self.h.clear()
        self.who.randomize()
        self.o.clear()

    def feed_forward(self, q):
        self.i = q
        self.h = self.wih.times(self.i)
        self.h.applyFunc(qm.sigmoid)
        self.o = self.who.times(self.h)
        self.o.applyFunc(qm.sigmoid)

        return self.o

    def train(self, qset, aset, epochs, learning_rate):
        # ----------------------------------------------		INITIALIZATION

        for epoch in range(epochs):
            # print("training on epoch: ", epoch)
            rand_choice = qm.np.random.randint(0, qset.rows)
            curr_q = qm.Matrix()
            curr_q.assign([qset.row(rand_choice)])

            curr_a = qm.Matrix()
            curr_a.assign([aset.row(rand_choice)])

            curr_a.transpose("replace")
            curr_q.transpose("replace")

            # curr_q.printValue("Question")
            # curr_a.printValue("Answer")

            guess = self.feed_forward(curr_q)

            error_o = curr_a.minus(guess)
            error_o.printValue("error")
            error_h = self.who.transpose().times(error_o)

            delta_who = error_o.times(self.h.transpose()).multiply(learning_rate)
            delta_wih = error_h.times(self.i.transpose()).multiply(learning_rate)

            self.who.addMatrix(delta_who)
            self.wih.addMatrix(delta_wih)


costs = []


class Custom_DFF:
    def __init__(self, layers, activation=qm.sigmoid, cost=qm.Losses.MSE):
        self.activation = activation
        self.layers = [None for a in range(len(layers))]
        for l in range(len(self.layers)):
            self.layers[l] = qm.clear_matrix(layers[l], 1)

        self.weights = [None for c in range(len(layers) - 1)]
        for w in range(len(self.weights)):
            self.weights[w] = qm.random_matrix(
                self.layers[w + 1].rows, self.layers[w].rows
            )

        self.biases = [None for d in range(len(layers))]
        for b in range(len(self.biases)):
            self.biases[b] = qm.clear_matrix(layers[b], 1)
        del self.biases[0]
        self.cost = cost

    def feed_forward(self, q):
        self.layers[0] = q
        for l in range(1, len(self.layers)):
            self.layers[l] = self.weights[l - 1].times(self.layers[l - 1])
            self.layers[l].addMatrix(self.biases[l - 1])
            self.layers[l].applyFunc(self.activation.function)
        return self.layers[-1]

    def printValues(self):
        for layer in self.layers:
            layer.printValue("layer")
        for weight in self.weights:
            weight.printValue("weight")
        for b in self.biases:
            b.printValue("bias")

    def backpropagate(self, error, learning_rate):
        errors = [None for i in range(len(self.layers) - 1)]
        errors[-1] = error
        # errors[-1].printValue("sis")

        for e in range(len(errors[:-1]))[::-1]:
            errors[e] = self.weights[e + 1].transpose().times(errors[e + 1])

        gradients = [None for i in range(len(self.layers) - 1)]

        for g in range(len(gradients)):
            gradients[g] = self.layers[g + 1].with_applied(self.activation.derivative)
            gradients[g] = gradients[g].dot(errors[g])
            gradients[g].multiply(learning_rate)
            print(gradients[g])
            # gradients[g].printValue("gradients")

        deltas_w = [None for i in range(len(self.weights))]
        deltas_b = [None for i in range(len(self.biases))]

        for d in range(len(deltas_w)):
            deltas_w[d] = gradients[d].times(self.layers[d].transpose())
            # deltas_w[d].printValue("deltas of" + str(d))

        for w in range(len(self.weights)):
            self.weights[w].addMatrix(deltas_w[w])

        for b in range(len(self.biases)):
            self.biases[b].addMatrix(gradients[b])

    def train(self, qset, aset, epochs, learning_rate):
        costs = []
        for epoch in range(epochs):
            # print(epoch, "/", epochs)
            rand_choice = qm.np.random.randint(0, qset.rows)
            curr_q = qm.Matrix()
            curr_q.assign([qset[rand_choice]])

            curr_a = qm.Matrix()
            curr_a.assign([aset[rand_choice]])

            curr_a.transpose("replace")
            curr_q.transpose("replace")

            # curr_q.printValue("question")
            # curr_a.printValue("answer")

            guess = self.feed_forward(curr_q)
            costs.append(self.cost.function(curr_a, guess))
            # costs.append(error_o)
            # guess.printValue("guess")
            # curr_a.printValue("aNS")
            # guess.printValue("ses")

            der_cost = self.cost.derivative(curr_a, guess)

            self.backpropagate(der_cost, learning_rate)
        # plt.plot(range(epochs), costs)
        # plt.show()


class Layers:
    class Dense:
        def __init__(self, neurons, activation, lastLayer):
            self.tag = "normal"
            self.values = qm.clear_matrix(neurons, 1)
            self.weights = qm.random_matrix(self.values.rows, lastLayer.values.rows)
            self.lastLayer = lastLayer
            self.activation = activation
            self.bias = qm.clear_matrix(neurons, 1)
            self.error = None

        def feed_forward(self):
            self.values = self.weights.times(self.lastLayer.values)
            self.values.addMatrix(self.bias)
            self.values.applyFunc(self.activation.function)

        def get_next_error(self):
            self.lastLayer.error = self.weights.transpose().times(self.error)

        def get_gradient(self):
            gradient = self.values.with_applied(self.activation.derivative)
            self.gradient = gradient.dot(self.error)
            return self.gradient

        def adjust(self):
            gradient = self.values.with_applied(self.activation.derivative)
            gradient = gradient.dot(self.error)
            gradient.multiply(0.01)

            delta_w = gradient.times(self.lastLayer.values.transpose())

            self.weights.addMatrix(delta_w)
            self.bias.addMatrix(gradient)

    class Input:
        def __init__(self, neurons, activation=qm.linear):
            self.values = qm.clear_matrix(neurons, 1)
            self.tag = "Input"
            self.error = None

        def get_next_error(self):
            pass

        def adjust(self):
            pass

    class Reshape:
        def __init__(self, rows, cols, activation, lastLayer):
            self.tag = "normal"
            self.values = qm.clear_matrix(rows, cols)
            self.weights = qm.random_matrix(len(self.values), lastLayer.values.rows)
            self.lastLayer = lastLayer
            self.activation = activation
            self.bias = qm.clear_matrix(rows, cols)
            self.error = None

        def feed_forward(self):
            self.values = self.weights.times(self.lastLayer.values)
            self.values.addMatrix(self.bias)
            self.values.applyFunc(self.activation.function)

        def get_next_error(self):
            self.lastLayer.error = self.weights.transpose().times(self.error)

        def adjust(self):
            gradient = self.values.with_applied(self.activation.derivative)
            gradient = gradient.dot(self.error)
            gradient.multiply(0.01)

            delta_w = gradient.times(self.lastLayer.values.transpose())

            self.weights.addMatrix(delta_w)
            self.bias.addMatrix(gradient)

    class Join:
        def __init__(self, neurons, activation, prev_layers):
            self.tag = "normal"
            self.values = qm.clear_matrix(neurons, 1)
            self.weights = []
            self.prev_layers = prev_layers
            for layer in prev_layers:
                self.weights.append(
                    qm.random_matrix(self.values.rows, layer.values.rows)
                )
            self.activation = activation
            self.bias = qm.clear_matrix(neurons, 1)
            self.error = None

        def feed_forward(self):
            for w in range(len(self.weights)):
                self.values.addMatrix(self.weights[w].times(self.prev_layers[w].values))
            self.values.addMatrix(self.bias)

            self.values.applyFunc(self.activation.function)

        def get_next_error(self):
            for l in range(len(self.prev_layers)):
                self.prev_layers[l].error = (
                    self.weights[l].transpose().times(self.error)
                )

        def adjust(self):
            deltas = []
            for w in range(len(self.weights)):
                gradient = self.values.with_applied(self.activation.derivative)
                gradient = gradient.dot(self.error)
                gradient.multiply(0.01)

                deltas.append(gradient.times(self.prev_layers[w].values.transpose()))
                self.weights[w].addMatrix(deltas[w])
            self.bias.addMatrix(gradient)

    class Output(Dense):

        def get_next_error(self, error):
            self.error = error
            self.lastLayer.error = self.weights.transpose().times(self.error)

        def adjust(self):
            gradient = self.values.with_applied(self.activation.derivative)
            gradient = gradient.dot(self.error)
            gradient.multiply(0.01)

            delta_w = gradient.times(self.lastLayer.values.transpose())

            self.weights.addMatrix(delta_w)
            self.bias.addMatrix(gradient)

    class Dropout(Dense):
        def __init__(self, neurons, activation, lastLayer, probability):
            self.tag = "normal"
            self.values = qm.clear_matrix(neurons, 1)
            self.weights = qm.random_matrix(self.values.rows, lastLayer.values.rows)
            self.lastLayer = lastLayer
            self.activation = activation
            self.bias = qm.clear_matrix(neurons, 1)
            self.error = None
            self.probability = probability

        def feed_forward(self):
            # print("feeding trough dropout")
            self.values = self.weights.times(self.lastLayer.values)
            self.values.addMatrix(self.bias)
            self.values.applyFunc(self.activation.function)
            for i in range(self.values.rows):
                x = random.randint(0, 100) / 100
                if x <= self.probability:
                    self.values.matrix[i] = [0]
            # print(self.values)


class Models:
    class Sequential:
        def __init__(self, layers, cost):
            self.layers = layers
            self.cost = cost

        def feed_forward(self, q):
            for layer in self.layers:
                if layer.tag == "Input":
                    layer.values.assign(q)

                if layer.tag == "normal":
                    layer.feed_forward()

                if layer.tag == "Output":
                    return layer.values
            return self.layers[-1].values

        def backpropagate(self, x0, yhat, y):
            deltas = [None for i in range(len(self.layers) - 1)]

            deltas.append(
                qm.from_numpy(
                    self.cost.derivative(qm.to_numpy(y), qm.to_numpy(yhat))
                    * self.layers[-1].values.with_applied(
                        self.layers[-1].activation.derivative
                    )
                )
            )

            for l in range(len(self.layers) - 1)[::-1]:
                deltas[l] = deltas[l + 1].times(x0).transpose()

        def train(self, qset, aset, epochs, learning_rate, batch_size=4):
            for epoch in range(epochs):
                for q in range(qset.rows):
                    # print(q)
                    curr_q = qm.Matrix(data=[qset[q]]).transpose()
                    curr_a = qm.Matrix(data=[aset[q]]).transpose()

                    guess = self.feed_forward(curr_q)

                    loss = self.cost.function(qm.to_numpy(guess), qm.to_numpy(curr_a))
                    # plt.scatter(epoch, loss)
                    # plt.pause(0.0005)

                    self.backpropagate(curr_q, guess, curr_a)

            # plt.show()


class Dataset:
    def __init__(self, name, inp_scale=1, out_scale=1):
        self.name = name
        self.questions = qm.Matrix()
        self.answers = qm.Matrix()

    def set_answers(self, answer):
        self.answers.assign(answer)

    def set_questions(self, question):
        self.questions.assign(question)


qset = qm.Matrix()
qset.assign([[1, 0], [1, 1], [0, 0], [0, 1]])

aset = qm.Matrix(data=[[1, 0, 0, 1]]).transpose()

q = qm.Matrix(data=[qset[1]])
a = qm.Matrix(data=[aset[1]])


def Custom_DFF_xor():
    brain = Custom_DFF((2, 10, 10, 10, 1), qm.Activations.sigmoid)
    brain.train(qset, aset, 10000, 0.1)

    q = qm.Matrix(data=[[1, 0]]).transpose()

    brain.feed_forward(q).printValue("Answer to 1,0")
    brain.feed_forward(qm.Matrix(data=[[1, 1]]).transpose()).printValue("Answer to 1,1")


def save(net, filename):
    file = open(filename, "wb")
    pickle.dump(net, file)
    file.close()


def load(filename):
    file = open(filename, "rb")
    net = pickle.load(file)
    file.close()
    return net


# 	-------------------------		A COUPLE OF EXAMPLES WITH NONLINEAR REGRESSION PROBLEMS SOLVED WITH THIS LIBRARY


def DFF_paridispari():
    brain = DFF_5L(1, 2, 3, 2, 2)

    q_set = qm.Matrix(data=[[1, 2, 3, 4, 5, 6]]).transpose()

    a_set = qm.Matrix(data=[[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]])

    brain.train(q_set, a_set, 1000, 0.01)
    r = brain.feed_forward(qm.Matrix(data=[[8]]), qm.sigmoid)
    r.printValue("r(not binarized)")
    qm.binarize(r)
    r.printValue("r")


def DFF_xor():
    brain = DFF_5L(2, 2, 2, 2, 1)
    xor_dataset = qm.Matrix(data=[[1, 0], [0, 1], [1, 1], [0, 0]])
    xor_answers = qm.Matrix(data=[[1, 1, 0, 0]]).transpose()

    brain.train(xor_dataset, xor_answers, 10000, 1)

    brain.feed_forward(qm.Matrix(data=[[0, 1]]).transpose(), qm.sigmoid).printValue(
        "answer to 0, 1"
    )
    brain.feed_forward(qm.Matrix(data=[[1, 1]]).transpose(), qm.sigmoid).printValue(
        "answer to 1, 1"
    )


def DFF_3L_xor():
    brain = Custom_DFF([2, 2, 1])
    xor_dataset = qm.Matrix(data=[[1, 0], [0, 1], [1, 1], [0, 0]])
    xor_answers = qm.Matrix(data=[[1, 1, 0, 0]]).transpose()

    brain.train(xor_dataset, xor_answers, 100000, 1)
    q = qm.Matrix(data=[[1, 0]]).transpose()

    brain.feed_forward(q).printValue("Answer to 1,0")
    brain.feed_forward(qm.Matrix(data=[[1, 1]]).transpose()).printValue("Answer to 1,1")

    while True:
        q = input()
        if q.split(",")[0] == "retrain":
            brain.train(xor_dataset, xor_answers, int(q.split(",")[1]), 1)
        elif q == "reset":
            brain = DFF_3L(2, 3, 1)
        elif q == "quit":
            quit()
        else:
            try:
                q = list(q.split(","))
                q = [q]
                q = qm.Matrix(data=q).transpose()
                q.applyFunc(float)
                q.printValue("ses")
                brain.feed_forward(q).printValue("sos")
            except ValueError:
                print("hai sbagliato comando")


def perc_exam():
    q_set = qm.Matrix(
        data=[
            [8, 2],
            [5, 5],
            [1, 9],
            [4, 6],
            [2, 8],
            [6, 4],
        ]
    )

    a_set = qm.Matrix(data=[[4], [7], [2], [8], [2], [9]])
    p = DFF_3L(2, 2, 1)
    p.train(q_set, a_set, 10000, 1)
    p.feed_forward(qm.Matrix(data=[[7], [3]])).printValue(
        "Il voto che prendi con 7h di studio e 3h di sonno"
    )


def perc_tabelline():
    p = Perceptron(1, 10)

    q_set = qm.Matrix(data=[[1, 2, 6, 4, 8]])
    q_set.transpose("replace")

    a_set = qm.Matrix(
        data=[
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            [6, 12, 18, 24, 30, 36, 42, 48, 54, 60],
            [4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
            [8, 16, 24, 32, 40, 48, 56, 64, 72, 80],
        ]
    )

    p.train(q_set, a_set, 1000, 0.01)

    p.feed_forward(qm.Matrix(data=[[5]])).printValue("tabllellina del 5")
