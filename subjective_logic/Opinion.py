"""
Opinion package
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

Package encompassing useful elements concerning the subjective logic opinions
"""


import math
from NotAnOpinionException import *
import mpmath
import numpy
import pylab
from config import epsilon
import sys
from beta_distribution.BetaDistribution import BetaDistribution

def get_random_opinion():
    """
    Static function for obtaining a random opinion compliant with 
    the subjective logic requirement
    """
    while True:
        r1 = mpmath.rand()
        r2 = mpmath.rand()
        if r1 + r2 < 1:
            return Opinion(r1, r2, mpmath.mpf("1") - (r1 + r2), "1/2")
        
def get_random_opinion_different(op):
    """
    Static function for obtaining a random opinion compliant with
    the subjective logic requirement which is different from the 
    opinion received as input
    """
    if isinstance(op, Opinion):
        while True:
            r = get_random_opinion()
            if r != op:
                #print >> sys.stderr, "Random opinion generated: " + repr(r)
                return r
    else:
        raise Exception("Opinion object expected")

def get_opinion_from_beta(alpha, beta, a = 1/2, W=2):
    rx = max(0, alpha - a * W)
    sx = max(0, beta - (1 - a) * W)
    return Opinion((rx / (rx + sx + W)), (sx / (rx + sx + W)), (W / (rx + sx + W)), a)

def get_opinion_from_list(l):
    return Opinion(l[0], l[1], l[2], l[3])
                

class Opinion():
    """
    Class representing a subjective opinion
    """
   
    def getBelief(self):
        return self._belief
    
    def getDisbelief(self):
        return self._disbelief
    
    def getUncertainty(self):
        return self._uncertainty
    
    def getBase(self):
        return self._base
    
    def __init__(self, b, d, u = None, a = None):
        self._belief = mpmath.mpf(b)
        self._disbelief = mpmath.mpf(d)

        if u is None:
            self._uncertainty = mpmath.mpf(1 - self._belief - self._disbelief)
        else:
            self._uncertainty = mpmath.mpf(u)

        if a is None:
            self._base = mpmath.mpf("1/2")
        else:
            self._base = mpmath.mpf(a)
        self.check()

    def check(self):
        """
        Method for checking if this object is compliant with the subjective logic
        constraints.
        """
        if not (
                (mpmath.almosteq(self._belief, mpmath.mpf("0"),epsilon) or mpmath.almosteq(self._belief, mpmath.mpf("1"),epsilon) or (self._belief >= 0 and self._belief <= 1)) and
                (mpmath.almosteq(self._disbelief, mpmath.mpf("0"),epsilon) or mpmath.almosteq(self._disbelief, mpmath.mpf("1"),epsilon) or (self._disbelief >= 0 and self._disbelief <= 1)) and
                (mpmath.almosteq(self._uncertainty, mpmath.mpf("0"),epsilon) or mpmath.almosteq(self._uncertainty, mpmath.mpf("1"),epsilon) or (self._uncertainty >= 0 and self._uncertainty <= 1)) and
                (mpmath.almosteq(self._base, mpmath.mpf("0"),epsilon) or mpmath.almosteq(self._base, mpmath.mpf("1"),epsilon) or (self._base >= 0 and self._base <= 1)) and
                (mpmath.almosteq(self._belief + self._disbelief + self._uncertainty, 1, epsilon))
            ):
            raise NotAnOpinionException(self)
        return True

    def __eq__(self, another):
        if (isinstance(another, Opinion)):
            return (mpmath.almosteq(self.getBelief(), another.getBelief(),epsilon) and \
                    mpmath.almosteq(self.getDisbelief(), another.getDisbelief(),epsilon) and \
                    mpmath.almosteq(self.getUncertainty(), another.getUncertainty(),epsilon) and\
                    mpmath.almosteq(self.getBase(), another.getBase(),epsilon)
                    )
        return NotImplemented

    def __ne__(self, another):
        result = self.__eq__(another)
        if result is NotImplemented:
            return result
        return not result

    def __repr__(self):
        return "<"+str(self._belief)+", "+str(self._disbelief)+", "+ str(self._uncertainty) +", " + str(self._base)+">"

    def get_x_cartesian(self):
        """
        @return: the x coordinate in the associated Cartesian plane
        """
        return (self._disbelief + self._uncertainty * mpmath.cos(mpmath.pi/3)) / mpmath.sin(mpmath.pi/3)

    def get_y_cartesian(self):
        """
        @return: the y coordinate in the associated Cartesian plane
        """
        return self._uncertainty

    def plot_basic(self):
        x = numpy.arange(0.0, float(1/mpmath.sin(mpmath.pi/3)), 0.01)
        t = []
        for l in numpy.nditer(x):
            if l < (1 / (2 * mpmath.sin(mpmath.pi/3))):
                t.append(float(mpmath.tan(mpmath.pi/3) * l))
            else:
                t.append(float(1 - mpmath.tan(mpmath.pi/3) * (l - 1/(2*mpmath.sin(mpmath.pi/3)))))

        pylab.plot(x, t)
        pylab.hold(True)


    def plot_vector(self):
        self.plot_basic()
        X,Y,U,V = zip(numpy.array([0,0,float(self.get_x_cartesian()), float(self.get_y_cartesian())]))
        pylab.quiver(X,Y,U,V,angles='xy',scale_units='xy',scale=1)
        pylab.show()

    def plot_point(self):
        self.plot_basic()
        pylab.plot(float(self.get_x_cartesian()), float(self.get_y_cartesian()), marker='o')
        pylab.show()

    def get_angle_alpha(self):
        if (mpmath.almosteq(self.getBelief(), 1, epsilon)):
            return mpmath.mpf("0")
        return mpmath.atan((self.getUncertainty() * mpmath.sin(mpmath.pi/mpmath.mpf("3"))) / (self.getDisbelief() + self.getUncertainty() * mpmath.cos(math.pi/mpmath.mpf("3"))))

    def get_angle_beta(self):
        if mpmath.almosteq(self.getDisbelief(), 1, epsilon):
            return mpmath.pi/3
        return mpmath.atan((self.getUncertainty() * mpmath.sin(mpmath.pi/mpmath.mpf("3"))) / (mpmath.mpf("1") - (self.getDisbelief() + self.getUncertainty() * mpmath.cos(mpmath.pi/mpmath.mpf("3")))))

    def get_angle_gamma(self):
        return ((mpmath.pi/3) - self.get_angle_beta())

    def get_angle_delta(self):
        if mpmath.almosteq(self.getUncertainty(), 1, epsilon):
            return mpmath.mpf("0")
        else:
            return mpmath.asin(self.getBelief() / self.get_length_to_uncertainty())

    def get_angle_epsilon(self):
        return (mpmath.pi - self.get_angle_gamma() - self.get_angle_delta())

    def get_length_to_uncertainty(self):
        return mpmath.sqrt( (mpmath.mpf("1/3") * \
                             mpmath.power((1 + self._disbelief - self._uncertainty), mpmath.mpf("2"))) + \
                           mpmath.power((self._belief), mpmath.mpf("2")))

    def get_max_x_cartesian(self):
        return (mpmath.mpf("2") - self.get_y_cartesian() + mpmath.tan(self.get_angle_alpha()) * self.get_x_cartesian()) / (mpmath.tan(self.get_angle_alpha()) + mpmath.sqrt("3"))

    def get_max_y_cartesian(self):
        return (- mpmath.sqrt("3") * self.get_x_cartesian()) + mpmath.mpf("2")

    def get_magnitude_ratio(self):
        return (mpmath.sqrt(mpmath.power(self.get_x_cartesian(), mpmath.mpf("2")) + \
                            mpmath.power(self.get_y_cartesian(), mpmath.mpf("2"))) / \
                (mpmath.sqrt(mpmath.power(self.get_max_x_cartesian(), mpmath.mpf("2")) \
                             + mpmath.power(self.get_max_y_cartesian(), mpmath.mpf("2")))))

    def expected_value(self):
        return self.getBelief() + self.getUncertainty() * self.getBase()


    def distance(self, another):
        """
        This method computes the Euclidean distance with another opinion and returns it
        """
        return mpmath.sqrt(mpmath.power(self.getBelief() - another.getBelief(), mpmath.mpf("2")) + \
                           mpmath.power(self.getDisbelief() - another.getDisbelief(), mpmath.mpf("2")) + \
                           mpmath.power(self.getUncertainty() - another.getUncertainty(), mpmath.mpf("2"))\
                           )

    def distance_expected_value(self, another):
        """
        This method computes the distance between the two expected values
        """
        return mpmath.absmax(self.expected_value() - another.expected_value())

    def get_beta_distribution_parameters(self, W = 2):
        prior = mpmath.mpf(W)

        uncertainty = self.getUncertainty()
        if self.getUncertainty() == 0:
            uncertainty = mpmath.mpf("1e-12")

        disbelief = self.getDisbelief()
        if self.getDisbelief() == 0:
            disbelief = mpmath.mpf("1e-12")

        belief = self.getBelief()
        if self.getBelief() == 0:
            belief = mpmath.mpf("1e-12")

        return [prior/uncertainty * self.getBelief() + prior * self.getBase(), prior/uncertainty * self.getDisbelief() + prior * (1-self.getBase())]

        #return [(1.0 + belief - disbelief)/uncertainty, (1.0 - belief + disbelief)/uncertainty]

    def getBetaDistribution(self):
        [alpha, beta] = self.get_beta_distribution_parameters()
        return BetaDistribution(alpha, beta)

    def toList(self):
        return [float(self.getBelief()), float(self.getDisbelief()), float(self.getUncertainty()), float(self.getBase())]

    def betaSum(self, y):
        if not isinstance(y, Opinion):
            raise NotAnOpinionException(y)

        return self._get_opinion_from_Beta(self.getBetaDistribution().sum(y.getBetaDistribution()))

    def betaUnion(self, y):
        if not isinstance(y, Opinion):
            raise NotAnOpinionException(y)

        return self._get_opinion_from_Beta(self.getBetaDistribution().union(y.getBetaDistribution()))

    def betaUnionMinus(self, y):
        if not isinstance(y, Opinion):
            raise NotAnOpinionException(y)

        return self._get_opinion_from_Beta(self.getBetaDistribution().unionMinus(y.getBetaDistribution()))

    def betaProduct(self, y):
        if not isinstance(y, Opinion):
            raise NotAnOpinionException(y)

        return self._get_opinion_from_Beta(self.getBetaDistribution().product(y.getBetaDistribution()))

    def betaDivision(self, y):
        if not isinstance(y, Opinion):
            raise NotAnOpinionException(y)

        return self._get_opinion_from_Beta(self.getBetaDistribution().division(y.getBetaDistribution()))


    def _get_opinion_from_Beta(self, Bdist, a = 1/2, W = 2):
        alpha = Bdist.getAlpha()
        beta = Bdist.getBeta()
        rx = max(0, alpha - a * W)
        sx = max(0, beta - (1 - a) * W)
        return Opinion((rx / (rx + sx + W)), (sx / (rx + sx + W)), (W / (rx + sx + W)), a)

    def plus(self, y):
        if not isinstance(y, Opinion):
            raise NotAnOpinionException(y)

        [b1,d1,u1,a1] = [self.getBelief(), self.getDisbelief(), self.getUncertainty(), self.getBase()]
        [b2,d2,u2,a2] = [y.getBelief(), y.getDisbelief(), y.getUncertainty(), y.getBase()]
        u = (a1 * u1 + a2 * u2) / (a1 + a2)
        d = max(0.0, (a1 * (d1 - b2) + a2 * (d2 - b1)) / (a1 + a2))
        #b = min(b1 + b2, 1.0)
        b = 1 - u - d
        a = min(a1 + a2, 1.0)
        return Opinion(b,d,u,a)

    def times(self, y):
        if not isinstance(y, Opinion):
            raise NotAnOpinionException(y)

        [b1, d1, u1, a1] = [self.getBelief(), self.getDisbelief(), self.getUncertainty(), self.getBase()]
        [b2, d2, u2, a2] = [y.getBelief(), y.getDisbelief(), y.getUncertainty(), y.getBase()]

        a = a1 * a2
        b = b1 * b2 + ((1 - a1) * a2 * b1 * u2 + a1 * (1 - a2) * u1 * b2) / (1 - a1 * a2)
        u = u1 * u2 + ((1 - a2) * b1 * u2 + (1 - a1) * u1 * b2) / (1 - a1 * a2)
        d = min(1, d1 + d2 - d1 * d2)
        return Opinion(b, d, u, a)

    def negate(self):
        return Opinion(self.getDisbelief(),self.getBelief(),self.getUncertainty(),1 - float(self.getBase()))

    def division(self, y):
        if not isinstance(y, Opinion):
            raise NotAnOpinionException(y)

        if float(y.getBelief()) == 1.0:
            return self

        [b1, d1, u1, a1] = [self.getBelief(), self.getDisbelief(), self.getUncertainty(), self.getBase()]
        [b2, d2, u2, a2] = [y.getBelief(), y.getDisbelief(), y.getUncertainty(), y.getBase()]

        e1 = b1 + u1*a1
        e2 = b2+ u2 * a2

        if not ((a1<=a2) and (d1>=d2) and (b1*(1-a1)*a2*(1-d2) >= a1*(1-a2)*(1-d1)*b2) and (u1*(1-a1)*(1-d2)>=u2*(1-a2)*(1-d1)) and a2!=0 ):
            return Opinion(0.0, 0.0, 1.0, 0.5)
        else:
            a = a1/a2
            b = 0.0
            d = 0.0
            u = 0.0
            if e1 == 0:
                d = 1.0
            elif a==1:
                b = 1.0
            else:
                e = e1 / e2
                d = min(max(0, (d1 - d2) / (1 - d2)), 1)
                u = min(max(0, (1 - d - e) / (1 - a)), 1)
                b = min(max(0, (1 - d - u)), 1)
            return Opinion(b,d,u,a)

