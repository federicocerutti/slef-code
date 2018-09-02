"""
BetaDistribution package
Copyright (c) 2013 Federico Cerutti <federico.cerutti@acm.org>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

DESCRIPTION:

Package for creating a beta distribution
"""

import math
from NotABetaDistributionException import *
import pulp

import scipy.special
import numpy
import pylab

class BetaDistribution():

    def getAlpha(self):
        return float(self._alpha)
    
    def getBeta(self):
        return float(self._beta)

    
    def __init__(self, alpha, beta):
        self._alpha = alpha
        self._beta = beta
        self.check()


    def check(self):
        if not(self._alpha > 0 and self._beta > 0):
            raise NotABetaDistributionException(self)

    def __repr__(self):
        return "BetaDistribution("+repr(self._alpha)+","+repr(self._beta)+")"


    def distribution(self, p):
        if not(p >= 0 and p <= 1):
            raise Exception("Distributions are computed between 0 and 1...")

        if (p == 0 and self._alpha < 1):
            return 0
        if (p == 1 and self._beta < 1):
            return 0

        return 1 / scipy.special.beta(self._alpha, self._beta)* \
                math.pow(p, (self._alpha - 1)) * math.pow((1 - p), (self._beta - 1))


    def plotdistribution(self):
        p = numpy.arange(0.0, 1.0, 0.01)
        r = []
        for a in numpy.nditer(p):
            r.append(self.distribution(a))

        pylab.plot(p, r)
        pylab.show()

    def mean(self):
        return float(self.getAlpha() / (self.getAlpha() + self.getBeta()))

    def variance(self):
        return float(self.getAlpha() * self.getBeta() / ( (self.getAlpha() + self.getBeta())**2  * (self.getAlpha() + self.getBeta() + 1) ))

    def _moment_matching(self, mean, var):

        sx = ((mean * (1 - mean)) / var - 1)

        # if mean * (1-mean) < 1e-2 and var < 1e-4:
        #     sx = 1.0 / var

        alpha = max(0.01, mean * sx)
        beta = max(0.01, (1 - mean) * sx)
        # alpha = max(0.1, mean * (mean - mean**2 - var) / var)
        # beta = max(0.1, (1.0 - mean) * (mean - mean**2 - var) / var )
        return [alpha, beta]

    def union(self, Y):
        if not isinstance(Y, BetaDistribution):
            raise NotABetaDistributionException(Y)

        mean = self.mean() + Y.mean() - self.mean() * Y.mean()
        var = self.variance() + 2 * Y.mean() * self.variance() + \
            Y.variance() + 2 * self.mean() * Y.variance() + \
            self.variance() * Y.variance() + self.variance() * (Y.mean())**2 + (self.mean())**2 * Y.variance()

        [alpha, beta] = self._moment_matching(mean, var)

        return BetaDistribution(alpha, beta)

    def unionMinus(self, Y):
        if not isinstance(Y, BetaDistribution):
            raise NotABetaDistributionException(Y)

        mean = self.mean() - Y.mean() + self.mean() * Y.mean()
        var = self.variance() + 2 * Y.mean() * self.variance() + \
            Y.variance() + 2 * self.mean() * Y.variance() + \
            self.variance() * Y.variance() + self.variance() * (Y.mean())**2 + (self.mean())**2 * Y.variance()

        [alpha, beta] = self._moment_matching(mean, var)

        return BetaDistribution(alpha, beta)

    def sum(self, Y):
        if not isinstance(Y, BetaDistribution):
            raise NotABetaDistributionException(Y)

        mean = self.mean() + Y.mean()
        var = self.variance() + Y.variance()

        [alpha, beta] = self._moment_matching(mean, var)

        return BetaDistribution(alpha, beta)

    def product(self, Y):
        if not isinstance(Y, BetaDistribution):
            raise NotABetaDistributionException(Y)

        mean = self.mean() * Y.mean()
        var = self.variance() * Y.variance() + \
              self.variance() * (Y.mean())**2 + Y.variance() * (self.mean()) ** 2

        [alpha, beta] = self._moment_matching(mean, var)

        return BetaDistribution(alpha, beta)

    def division(self, Y):
        if not isinstance(Y, BetaDistribution):
            raise NotABetaDistributionException(Y)

        # if self.mean() > Y.mean():
        #     print("%s / %s " % (str(self), str(Y)))
        #     return BetaDistribution(1,1)

        # mean = max(1e-6, min(1.0-1e-6, scipy.special.beta(self.getAlpha() + 1, self.getBeta()) * scipy.special.beta(Y.getAlpha() - 1, Y.getBeta()) / \
        #         (scipy.special.beta(self.getAlpha(), self.getBeta()) * scipy.special.beta(Y.getAlpha(), Y.getBeta()))))

        mean = min(1.0-1e-6, self.mean() / Y.mean())

        var = (self.variance() + self.mean()**2) * (Y.variance()+Y.mean()**2)/(Y.mean()**4) - (self.mean()/Y.mean())**2

        # var = scipy.special.beta(self.getAlpha() + 2, self.getBeta()) * scipy.special.beta(Y.getAlpha() - 2, Y.getBeta()) / \
        #        (scipy.special.beta(self.getAlpha(), self.getBeta()) * scipy.special.beta(Y.getAlpha(), Y.getBeta()))

        var = min(var, mean ** 2 * (1.0 - mean) / (1.0 + mean), (1.0 - mean) ** 2 * mean / (2 - mean))

        [alpha, beta] = self._moment_matching(mean, var)

        # sx = mean * (1.0-mean) / var - 1
        #
        # if mean == 0.0 or mean == 1.0:
        #     sx = 1e10
        # else:
        #     sx = max(sx, 1.0/mean, 1.0/(1.0-mean))
        #
        #
        # alpha = 0.0
        # beta = 0.0
        #
        # if mean == 0.0:
        #     [alpha, beta] = [1, sx]
        # elif mean == 1.0:
        #     [alpha, beta] = [sx, 1]
        # else:
        #     [alpha, beta] = [mean * sx, (1.0 - mean) * sx]


        # if (alpha < 0.1 or beta < 0.1):
        #
        #     #if self.variance() < 0.1 or Y.variance() <0.1:
        #         #print("Uncertain Beta. (%s, %s) / (%s, %s) = (Mean: %s, Var: %s) " % (str(self.mean()), str(self.variance()), str(Y.mean()), str(Y.variance()), mean, var))
        #         #print("Ratio of means: %s" % str(float(self.mean())/float(Y.mean())))
        #
        #     mean = float(self.mean())/ float(Y.mean())
        #     # var = float(Y.variance())
        #     #
        #     # [alpha, beta] = self._moment_matching(mean, var)
        #     #
        #     # if (alpha < 0.1 or beta < 0.1):
        #     #     print("Uncertain Beta. (%s, %s) / (%s, %s) = (Mean: %s, Var: %s) " % (
        #     #     str(self.mean()), str(self.variance()), str(Y.mean()), str(Y.variance()), mean, var))
        #     #     print("Ratio of means: %s" % str(float(self.mean())/float(Y.mean())))
        #
        #     #
        #     # uncertainty = 0.0
        #     # belief = 0.0
        #     # lp = pulp.LpProblem("Max Uncertainty", pulp.LpMaximize)
        #     # u = pulp.LpVariable('b', lowBound=0.0, upBound=1.0, cat='Continuous')
        #     # b = pulp.LpVariable('u', lowBound=0.0, upBound=1.0, cat='Continuous')
        #     # lp += 1 * u, "Z"
        #     # lp += b + 0.5 * u <= mean
        #     # lp += b + 0.5 * u >= mean
        #     # lp += b + u <= 1.0
        #     # lp.solve()
        #     #
        #     # valueu = 0.0
        #     # valueb = 0.0
        #     #
        #     # for variable in lp.variables():
        #     #     if variable.name == 'b':
        #     #         valueb = variable.varValue
        #     #     elif variable.name == 'u':
        #     #         valueu = variable.varValue
        #     #
        #     # if valueu == 0:
        #     #     valueu = 0.0000001
        #     #
        #     valueu = 0.0
        #     valueb = 0.0
        #     if 0 < mean <= 0.5:
        #         valueu = mean * 2
        #     elif 0.5 < mean <= 0.666:
        #         uncertainty = 0.0
        #         belief = 0.0
        #         lp = pulp.LpProblem("Max Uncertainty", pulp.LpMaximize)
        #         u = pulp.LpVariable('b', lowBound=0.0, upBound=1.0, cat='Continuous')
        #         b = pulp.LpVariable('u', lowBound=0.0, upBound=1.0, cat='Continuous')
        #         lp += 1 * u, "Z"
        #         lp += b + 0.5 * u <= mean
        #         lp += b + 0.5 * u >= mean
        #         lp += b + u <= 1.0
        #         lp.solve()
        #
        #         for variable in lp.variables():
        #             if variable.name == 'b':
        #                 valueb = variable.varValue
        #             elif variable.name == 'u':
        #                 valueu = variable.varValue
        #     elif 0.66 < mean <= 1.0:
        #         lp = pulp.LpProblem("Max Uncertainty", pulp.LpMinimize)
        #         u = pulp.LpVariable('b', lowBound=0.0, upBound=1.0, cat='Continuous')
        #         b = pulp.LpVariable('u', lowBound=0.0, upBound=1.0, cat='Continuous')
        #         lp += 1 * u, "Z"
        #         lp += b + 0.5 * u <= mean
        #         lp += b + 0.5 * u >= mean
        #         lp += b + u <= 1.0
        #         lp.solve()
        #
        #         for variable in lp.variables():
        #             if variable.name == 'b':
        #                 valueb = variable.varValue
        #             elif variable.name == 'u':
        #                 valueu = variable.varValue
        #
        #     alpha = 2.0/valueu * valueb + 1
        #     beta = 2.0/valueu * (1 - valueb - valueu) + 1

        return BetaDistribution(alpha, beta)