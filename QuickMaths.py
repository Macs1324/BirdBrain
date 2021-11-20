import random, math
import numpy as np
import time

#A python list-based 2D matrix library made by my 15 year old self that I've been reusing for years
#numpy is for noobs
#Comments are in italian

class Matrix:
    #il metodo costruttore della matrice, che la inizializza con un determinato numero di righe e colonne
    def __init__(self, rows = 1, cols = 1, data=None):
        if data != None:
            self.rows = len(data)
            self.cols = len(data[0])
            self.assign(data)
        else:
            self.rows = rows
            self.cols = cols
            self.matrix = [[None for i in range(cols)] for j in range(rows)]
    def shape(self):
        return [self.rows, self.cols]

    #Funzione per mandare in output i valori della matrice, aggiungendo un commento facoltativo
    def printValue(self, comment = ""):
        r = ""
        r += comment + "\n"
        for i in range(self.rows):
            for j in range(self.cols):
                r += str(self.matrix[i][j])
                r += ", "
            r += "\n"
        print(r)
        return r

    #Funzione per restituire una determinata riga della matrice
    def row(self, index):
        return self.matrix[index]

    #Funzione per restituire una determinata colonna della matrice
    def col(self, index):
        for i in range(self.rows):
            r = []
            for i in range(self.rows):
                r.append(self.matrix[i][index])
            return r

    def randomize(self, minval = -100, maxval = 100):
        #random.seed(1)
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] = random.randint(minval, maxval) / 100


    def clear(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] = 0


    def assign(self, inp):
        self.matrix = inp
        self.rows = len(inp)
        self.cols = len(inp[0])

    #Funzione che permette di trasporre la matrice da verticale a orizzontale e viceversa
    def transpose(self, repl = "x"):
        r = Matrix(self.cols, self.rows)
        for i in range(r.rows):
            r.matrix[i] = self.col(i)
        if repl == "replace":
            self.matrix = r.matrix
            self.cols = r.cols
            self.rows = r.rows
        else:
            return r

    #Moltiplicazione tra vettori


    #Moltiplicazione tra matrici (malfunzionante)
    def times(self, m2):
        if self.cols == m2.rows:
            r = Matrix(self.rows, m2.cols)
            r.clear()
            for i in range(r.rows):
                for j in range(r.cols):
                    r.matrix[i][j] = vectorDot(self.row(i), m2.col(j))
            return r
        else:
            print("Matrices not aligned")

    #Incrementa tutti gli elementi della matrice
    def increment(self, amount):
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] += amount

    #Funzione per cambiare la forma della matrice (malfunzionante)
    def reshape(self, rowz, colz):
        if rowz * colz == self.rows * self.cols:
            r = Matrix(rowz, colz)
            temp = []
            for i in range(self.rows):
                for j in range(self.cols):
                    temp.append(self.matrix[i][j])
            rowInd = 0
            colInd = 0
            for x in range(len(temp)):
                r.matrix[rowInd][colInd] = temp[x]
                colInd += 1
                if colInd >= r.cols:
                    colInd = 0
                    rowInd += 1
            r.rows = rowz
            r.cols = colz
            return r

        else:
            print("Error \n Different number of elements")

    def applyFunc(self, func):
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] = func(self.matrix[i][j])

    def plus(self, m2):
        r = Matrix(self.rows, self.cols)
        if self.rows == m2.rows and self.cols == m2.cols:
            for i in range(self.rows):
                for j in range(self.cols):
                    r.matrix[i][j] = self.matrix[i][j] + m2.matrix[i][j]
            return r
        else:
            print("Matrices not aligned")

    def minus(self, m2):
        r = Matrix(self.rows, self.cols)
        if self.rows == m2.rows and self.cols == m2.cols:
            for i in range(self.rows):
                for j in range(self.cols):
                    r.matrix[i][j] = self.matrix[i][j] - m2.matrix[i][j]
            return r
        else:
            print("Matrices not aligned")

    def multiply(self, amount):
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] *= amount
        return self

    def dot(self, amount):
        r = Matrix(self.rows, self.cols)
        r.assign(self.matrix)
        for i in range(self.rows):
            for j in range(self.cols):
                r.matrix[i][j] *= amount.matrix[i][j]
        return r

    def addMatrix(self, otherMatrix):
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] += otherMatrix.matrix[i][j]
    def __len__(self):
        return self.rows * self.cols

    def __dsigmoid__(self):
        r = Matrix()
        r.assign(self.matrix)
        r.applyFunc(dsigmoid)

        return r

    def __print__(self):
        self.printValue()

    def with_applied(self, func):
        r = Matrix(data=self.matrix)
        for i in range(r.rows):
            for j in range(r.cols):
                r.matrix[i][j] = func(r.matrix[i][j])
        return r

    def append(self, _type, _list):
        r = Matrix()
        if _type == "row":
            self.matrix.append(_list)
            self.rows += 1
        elif type == "col":
            for row in range(self.rows):
                r.matrix[row].append(_list[row])
            self.cols += 1

    def __getitem__(self, item):
        return self.matrix[item]

    def __str__(self):
        r = "\n"
        for i in range(self.rows):
            for j in range(self.cols):
                r += str(self[i][j])
                r += ", "
            r += "\n"
        return r

    def get_max_index(self):
        ind = 0
        m = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if self[i][j] > m:
                    m = self[i][j]
                    ind = [i, j]
        return ind

    def append_matrix(self, m, axis):
        if axis == "vertical":
            for i in range(m.rows):
                self.append("row", m.row(i))
        elif axis == "horizontal":
            for i in range(m.cols):
                self.append("col", m.col(i))

    def __add__(self, other):
        if other.__class__.__name__ == 'Matrix':
            r = Matrix()
            r.assign(self.plus(other).matrix)
            return r

        else:
            r = Matrix(data=self.matrix)
            r.increment(other)
            return r

    def __sub__(self, other):
        if other.__class__.__name__ == 'Matrix':
            r = Matrix()
            r.assign(self.minus(other).matrix)
            return r

        else:
            r = Matrix(data=self.matrix)
            for i in range(r.rows):
                for j in range(r.cols):
                    r[i][j] -= other
            return r

    def __mul__(self, other):
        if other.__class__.__name__ == "Matrix":
            r = Matrix(data=self.matrix)
            r = r.dot(other)

            return r
        else:
            r = Matrix(data=self.matrix)
            r.multiply(other)

            return r

    def __truediv__(self, other):
        if other.__class__.__name__ == 'Matrix':
            r = Matrix(data=self)
            for i in range(r.rows):
                for j in range(r.cols):
                    r[i][j] /= other[i][j]
            return r

        else:
            r = Matrix(data=self.matrix)
            for i in range(r.rows):
                for j in range(r.cols):
                    r[i][j] /= other
            return r




####################################################################################################################

class Tensor:
    def __init__(self):
        print("initialized tensor")




####################################################################################################################

			

class Activations:
    class sigmoid:
        @staticmethod
        def function(gamma):
            if gamma < 0:
                return 1 - 1/(1 + math.exp(gamma))
            else:
                return 1/(1 + math.exp(-gamma))

        @staticmethod
        def derivative(x):
            return x * (1 - x)

    class reLu:
        @staticmethod
        def function(x):
            return max(0, x)

        @staticmethod
        def derivative(x):
            if x <= 0:
                return 0
            elif x > 0:
                return 1

########################################################################


class Losses:
    class CrossEntropy:
        @staticmethod
        def function(yhat, y):
            y = to_numpy(y)
            yhat = to_numpy(yhat)
            eps = 1e-8
            return from_numpy(-(yhat*np.log(y+eps) + (1-yhat)*np.log(1-y+eps)))

        def derivative(yhat, y):
            y = to_numpy(y)
            yhat = to_numpy(yhat)
            eps = 1e-8
            return from_numpy((1-yhat)/(1-y+eps) - yhat/(y+eps))

    class MSE:
        @staticmethod
        def function(yHat, y):
            return sum((yHat - y) * (yHat - y))
        @staticmethod
        def derivative(yHat, y):
            return (yHat - y) * 2



			
def vectorDot(vec1, vec2):
    r = 0
    for i in range(len(vec1)):
        r += vec1[i] * vec2[i]
    return r

def sigmoid(gamma):
  if gamma < 0:
    return 1 - 1/(1 + math.exp(gamma))
  else:
    return 1/(1 + math.exp(-gamma))

def reLu(x):
    if x > 0:
        return x
    else:
        return 0

def linear(x):
    return x

def tanh(x):
    return math.tanh(x)

def leaky_reLu(x, alpha = 0.01):
    if x >= 0:
        return x
    else:
        return x * alpha * -1

def cost(x):
    r = 0
    for i in range(x.rows):
        for j in range(x.cols):
            r += x.matrix[i][j] ** 2
    return abs(r)

def sum(x):
    r = 0
    for i in range(x.rows):
        for j in range(x.cols):
            r += x.matrix[i][j]
    return r

def dsigmoid(x):
    return x * (1 - x)


def to_numpy(matrix):
	return np.array(matrix.matrix)
def from_numpy(matrix):
    shape = (len(matrix), len(matrix[0]))
    r = Matrix(shape[0], shape[1])
    for i in range(r.rows):
        for j in range(r.cols):
            r.matrix[i][j] = matrix[i][j]
    return r

def mean_sq_err(matrix):
    r = sum(matrix)
    r /= len(matrix)
    r *= r

    return r

def matrix_sum(matrix1, matrix2):
    r = Matrix(matrix1.rows, matrix1.cols)
    for i in range(matrix1.rows):
        for j in range(matrix1.cols):
            r.matrix[i][j] = matrix1.matrix[i][j] + matrix2.matrix[i][j]
    return r


def matmul(matrix1, matrix2):
    return Matrix(data=matrix1.times(matrix2).matrix)

def binarize(matrix, point=0.5):
    for i in range(matrix.rows):
        for j in range(matrix.cols):
            if matrix.matrix[i][j] < point:
                matrix.matrix[i][j] = 0
            else:
                matrix.matrix[i][j] = 1

def flatten_matrix(m):
    r = Matrix(1, m.rows * m.cols)
    r.assign(m.reshape(1, r.cols).matrix)
    return r

def tensor2matrix(t):
    r = Matrix(t[0].rows, t[0].cols * len(t))
    for i in range(t[0].rows):
        for j in range(t[0].cols):
            for m in range(len(t)):
                r[i][j] = t[m][i][j]
    return r


def flatten_tensor(t):
    m = tensor2matrix(t)
    m.printValue("sos")
    r = flatten_matrix(m)
    return r


def clear_matrix(r, c):
    r = Matrix(r, c)
    r.clear()
    return r

def random_matrix(r, c):
    r = Matrix(r, c)
    r.randomize(-100, 100)
    return r

def filled_matrix(r, c, par):
    r = Matrix(r, c)
    for i in range(r.rows):
        for j in range(r.cols):
            r[i][j] = par

    return r
