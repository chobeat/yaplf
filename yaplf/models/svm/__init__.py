r"""
Package handling support vector-based models in yaplf

Package yaplf.models.svm contains all the classes handling SV-based models in
yaplf.

TODO:

- Regression
- Quality-based SVC
- Quality-based SVR
- pep8 checked
- pylint score: 9.22

AUTHORS:

- Dario Malchiodi (2010-02-15): initial version

"""

# *****************************************************************************
#       Copyright (C) 2010 Dario Malchiodi <malchiodi@di.unimi.it>
#
# This file is part of yaplf.
# yaplf is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 2.1 of the License, or (at your option)
# any later version.
# yaplf is distributed in the hope that it will be useful, but without any
# warranty; without even the implied warranty of merchantability or fitness
# for a particular purpose. See the GNU Lesser General Public License for
# more details.
# You should have received a copy of the GNU Lesser General Public License
# along with yaplf; if not, see <http://www.gnu.org/licenses/>.
#
#*****************************************************************************


from yaplf.models.kernel import LinearKernel
from numpy import sign, dot, mean
from numpy import arange, array, ones, linalg
from pylab import plot, show
from sklearn import linear_model
from yaplf.models import Classifier
from yaplf.models.kernel import Kernel
from yaplf.data import LabeledExample
import numpy.linalg
from yaplf.utility.montecarlo import MonteCarloSimulator
import math
import sklearn.svm

def check_svm_classification_sample(sample):
    r"""
    Checks whether the supplied sample is properly formatted in order to use it
    for SVM classification, raising a specialized ValueError otherwise. The
    performed check requires that all patterns have the same dimension, while
    labels should be either set to 1 or to -1.

    INPUT:

    - sample -- iterable containing a sample to be checked.

    OUTPUT:

    No output. Raises a ValueError if the sample is not suitable to build a SVM
    classifier, otherwise returns silently.

    EXAMPLES:

    Any sample given as input to ``check_svm_classification_sample`` is checked
    against the following properties:

    - the sample should be iterable;

    - each sample element should have a ``pattern`` and a ``label`` field, that
      is it should be an instance of the ``LabeledExample`` class;

    - ``pattern`` fields of all sample elements should have the same dimension;

    - ``label`` fields of all sample elements should either be equal to `1` or
      to `-1`.

    Patterns of unequal length in a sample cause a ValueError to be thrown:

    ::
        >>> from yaplf.models.svm import check_svm_classification_sample
        >>> from yaplf.data import LabeledExample
        >>> wrong_sample = (LabeledExample((1, 0), 1),
        ... LabeledExample((1, 0, 1), 1))
        >>> check_svm_classification_sample(wrong_sample)
        Traceback (most recent call last):
            ...
        ValueError: SVM classification patterns should have the same dimension

    The same error is thrown, although with a different message, when any label
    is neither set to `1` or `-1`:

    ::

        >>> xor_sample = (LabeledExample((0., 0.), 0.),
        ... LabeledExample((1., 0.), 1.), LabeledExample((0., 1.), 1.),
        ... LabeledExample((1., 1.), 0.))
        >>> check_svm_classification_sample(xor_sample)
        Traceback (most recent call last):
            ...
        ValueError: SVM classification labels should be set either to -1 or to
        1

    When all properties are met the function returns silently

    ::

        >>> xor_sample = (LabeledExample((0., 0.), -1),
        ... LabeledExample((1., 0.), 1), LabeledExample((0., 1.), 1),
        ... LabeledExample((1., 1.), -1))
        >>> check_svm_classification_sample(xor_sample)

    Note that equality w.r.t. `1` and `-1` is type insensitive, that is ``1.0``
    and ``-1.0`` are legal values, too:

    ::

        >>> xor_sample = (LabeledExample((0., 0.), -1),
        ... LabeledExample((1., 0.), 1.0), LabeledExample((0., 1.), 1),
        ... LabeledExample((1., 1.), -1.0))
        >>> check_svm_classification_sample(xor_sample)

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

    """
    if len(sample) < 1:
        raise ValueError("Empty sample")
    dim = len(sample[0].pattern)

    if abs(sample[0].label) != 1:
        raise ValueError('SVM classification labels should be set either to \
-1 or to 1')

    for elem in sample[1:]:
        if len(elem.pattern) != dim:
            raise ValueError('SVM classification patterns should have the \
same dimension')
        if abs(elem.label) != 1:
            raise ValueError('SVM classification labels should be set either \
to -1 or to 1')


def check_svm_classification_unlabeled_sample(unlabeled_sample):
    if len(unlabeled_sample) < 1:
        return

    dim = len(unlabeled_sample[0])
    for elem in unlabeled_sample[1:]:
        if len(elem) != dim:
            raise ValueError('SVM classification patterns should have the \
same dimension')


class SVMClassifier(Classifier):
    r"""
    Class implementing the Support Vector Classifier (SVC for short), in the
    version originally introduced by [Cortes and Vapnik, 1995]. This model
    depends inherently on a subset of a given sample `\left\{ (x_1, y_1),
    \dots, (x_m, y_m) \right\} \subset X \cup \{ -1, 1 \}`, where `X` is a
    suitable space, as well as on set of *weights* `\left\{ \alpha_1, \dots,
    \alpha_m \right\}` (one weight for each sample item), on a *threshold `b
    \in \mathbb R`, and on a *kernel function* `k: X^2 \mapsto \mathbb R`.
    Precisely, when presented a generic pattern `x \in X` it outputs the sign
    of `\sum_{i=1}^m \alpha_i y_i k(x_i, x) + b`.

    INPUT:

    - ``alpha`` -- iterable containing the SVC weights.

    - ``threshold`` -- number containing the SVC threshold.

    - ``sample`` -- iterable containing the examples `(x_i, y_i)`.

    - ``kernel`` -- Kernel instance (default: LinearKernel()) SVC kernel
      function.

    OUTPUT:

    Classifier -- a SVMClassifier instance.

    EXAMPLES:

    ``SVMClassifier`` instances can be defined directly through the class
    constructor; for instance it is possible to get such a classifier for the
    binary AND sample:

    ::

        >>> from yaplf.data import LabeledExample
        >>> from yaplf.models.svm import SVMClassifier
        >>> and_sample = (LabeledExample((0., 0.), -1.),
        ... LabeledExample((1., 0.),1.), LabeledExample((0., 1.), 1.),
        ... LabeledExample((1., 1.), 1.))
        >>> svc = SVMClassifier((1, 1, 1, 0), -0.5, and_sample)

   To verify how ``svc`` correctly classifies ``and_sample`` it is possible to
   invoke the model's ``classify`` function on every pattern, subsequently
   comparing the result with the original labels:

    ::

        >>> map(svc.compute, [ e.pattern for e in and_sample ])
        [-1.0, 1.0, 1.0, 1.0]
        >>> [e.label for e in and_sample]
        [-1.0..., 1.0..., 1.0..., 1.0...]
        >>> map(svc.compute, [e.pattern for e in and_sample]) == \
        ... [e.label for e in and_sample]
        True

    The same result can be obtained more quickly, directly invoking the
    ``test`` model inherited by the ``Model`` base class:

    ::

        >>> svc.test(and_sample)
        0.0

    Specification of a kernel function allows more flexible SV-classifiers,
    able to correctly classify a more complex sample such as the binary XOR
    one:

    ::

        >>> from yaplf.models.kernel import GaussianKernel
        >>> xor_sample = (LabeledExample((0., 0.), -1.),
        ... LabeledExample((1., 0.), 1.), LabeledExample((0., 1.), 1.),
        ... LabeledExample((1., 1.), -1.))
        >>> svc = SVMClassifier([1.52, 2.02, 2.02, 1.52], -0.39, xor_sample,
        ... kernel = GaussianKernel(0.6))
        >>> map(svc.compute, [e.pattern for e in xor_sample])
        [-1.0, 1.0, 1.0, -1.0]
        >>> [e.label for e in xor_sample]
        [-1.0..., 1.0..., 1.0..., -1.0...]
        >>> svc.test(xor_sample)
        0.0

    Another way to figure out how a SV classifier behaves is through a plot of
    its decision function:

    ::

        >>> svc.plot((-0.5, 1.5), (-0.5, 1.5), margin = True, separator = True,
        ... shading = True, margin_color = 'red', margin_width = 7)

    For a more detailed view of how decision function plot can be fine tuned,
    see the documentation for ``plot`` function later on in this class.

    IMPLEMENTATION DEPENDENT ISSUES:

    When used withing sage, ``plot`` outputs a graphic object which is
    directly shown within a netbook or through a helper application if the
    invocation is made in a command-line interface.

    When used in a pure python environment ``plot`` outputs a matplotlib
    figure, which is opened or saved through the standard library functions;
    for instance, the following example draws the same plot of before and saves
    it in a file named ``svc-decision-function.png``:

    ::

        >>> from yaplf.graph import MatplotlibPlotter
        >>> fig = svc.plot((-0.5, 1.5), (-0.5, 1.5), margin = True,
        ... separator = True, shading = True, margin_color = 'red',
        ... margin_width = 7, plotter = MatplotlibPlotter())
        >>> fig.savefig('svc-decision-function.png')

    REFERENCES:

    [Cortes and Vapnik, 1995] Corinna Cortes and Vladimir Vapnik,
    Support-Vector Networks, Machine Learning 20 (1995), 273--297.

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """


    def overHardMargin(self, x):
        return (not self.c) or x < self.c

    def __init__(self, alpha, c, sample, kernel=LinearKernel(), **kwargs):
        r"""See ``SVMClassifier`` for full documentation.

        """

        Classifier.__init__(self)
        self.c = c
        num_patterns = len(sample)
        check_svm_classification_sample(sample)
        self.dim = len(sample[0].pattern)

        if len(alpha) != num_patterns:
            raise ValueError('The supplied sample and multipliers vector do \
not have the same size')
        self.kernel = kernel
        self.sv_indices = [i for i in range(len(alpha)) if alpha[i] != 0]

        self.support_vectors = [sample[i].pattern for i in self.sv_indices]
        self.signed_alphas = [alpha[i] * sample[i].label
                              for i in self.sv_indices]

        self.threshold = mean([sample[i].label -
                               sum([alpha[j] * sample[j].label *
                                    self.kernel.compute(sample[j].pattern,
                                                        sample[i].pattern) for j in range(num_patterns)])
                               for i in range(num_patterns)
                               if alpha[i] > 0 and self.overHardMargin(alpha[i])])


    def __repr__(self):
        alpha = [abs(a) for a in self.signed_alphas]
        # was
        # map(abs, self.signed_alphas)
        patterns = self.support_vectors
        labels = [sign(a) for a in self.signed_alphas]
        # was
        # labels = map(sign, self.signed_alphas)
        sample = [LabeledExample(*pl) for pl in zip(patterns, labels)]
        # was
        # sample = map(lambda x: LabeledExample(*x), zip(patterns, labels))

        result = 'SVMClassifier(' + str(alpha) + ', '
        result += str(self.threshold) + ', '
        result += str(sample.__repr__())
        if self.kernel != Kernel.get_default():
            result += ', kernel = ' + str(self.kernel.__repr__())
        result += ')'
        return result

    def __str__(self):
        return self.__repr__()

    def decision_function(self, pattern):
        r"""
        Returns the decision function associated by the classifier to the
        specified pattern. Its sign determines the class associated to the
        pattern.

        INPUT:

        - ``self`` -- SVMClassifier object on which the function is invoked.

        - ``pattern`` -- pattern on which the decision function is to be
          computed.

        OUTPUT:

        Number -- the SV classifier decision function value.

        EXAMPLES:

        This function is called in order to get the decision function value
        corresponding to a given pattern:

        ::

            >>> from yaplf.data import LabeledExample
            >>> from yaplf.models.svm import SVMClassifier
            >>> from yaplf.models.kernel import GaussianKernel
            >>> xor_sample = (LabeledExample((0., 0.), -1.),
            ... LabeledExample((1., 0.), 1.), LabeledExample((0., 1.), 1.),
            ... LabeledExample((1., 1.), -1.))
            >>> svc = SVMClassifier([1.52, 2.02, 2.02, 1.52], -0.39,
            ... xor_sample, kernel = GaussianKernel(0.6))
            >>> svc.decision_function((0, 0))
            -0.99712539305333991
            >>> svc.decision_function((1, 0))
            0.99756586384169432
            >>> svc.decision_function((0.5, 0))
            0.05142629338510496

        It is easy to see that che class in output of the SV classifier equals
        the sign of the corresponding decision function:

        ::

            >>> svc.compute((0, 0))
            -1.0
            >>> svc.compute((1, 0))
            1.0
            >>> svc.compute((0.5, 0))
            1.0

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        if len(pattern) != self.dim:
            raise ValueError('The supplied pattern is incompatible with the \
SVM dimension')

        kernel_values = [self.kernel.compute(x, pattern)
                         for x in self.support_vectors]
        # was
        # kernel_values = map(lambda x: self.kernel.compute(x, pattern),
        #     self.support_vectors)

        return dot(self.signed_alphas, kernel_values) + self.threshold

    def compute(self, pattern):
        r"""
        Associates a class (through values -1 and 1) to the specified
        pattern, computing the sign of the corresponding decision function
        value.

        INPUT:

        - ``self`` -- SVMClassifier object on which the function is invoked.

        - ``pattern`` -- pattern whose corresponding class is to be computed.

        OUTPUT:

        -1 or 1 -- the class associated to the supplied pattern by the SV
        classifier.

        EXAMPLES:

        Consider the following ``SVMClassifier`` instance expressly tailored in
        order to deal with the binary AND sample: cycling through the sample
        items and feeding their patterns to the ``compute`` method one obtains
        the original labels:

        ::

            >>> from yaplf.data import LabeledExample
            >>> from yaplf.models.svm import SVMClassifier
            >>> and_sample = (LabeledExample((0., 0.), -1.),
            ... LabeledExample((1., 0.),1.), LabeledExample((0., 1.), 1.),
            ... LabeledExample((1., 1.), 1.))
            >>> svc = SVMClassifier((1, 1, 1, 0), -0.5, and_sample)
            >>> map(svc.compute, [ e.pattern for e in and_sample ])
            [-1.0, 1.0, 1.0, 1.0]

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        return sign(self.decision_function(pattern))


class ESVMClassifier(Classifier):

    r"""

    INPUT:

    - ``solution`` -- list of iterables containing the solution of the E-SVM problem.

    - ``sample`` -- iterable containing the examples `(x_i, y_i)`.

    - ``unlabeled_sample`` -- iterable containing the unlabeled examples .

    - ``c`` -- weighting parameter for the minimization of the misclassification on labeled data

    - ``d`` -- weighting parameter for the minimization of the misclassification on unlabeled data

    - ``tube_tolerance`` -- tolerance on the evalution of the distance between a point and the tube margin

    - ``estimation_tolerance`` -- tolerance on the estimation of the tube radius and the threshold

    - ``kernel`` -- Kernel instance (default: LinearKernel()) SVC kernel
      function.

    - ``regrKernel`` -- Kernel for the regressor (default: "linear"). It must be one of 'linear', 'poly', 'rbf',
    'sigmoid','precomputed' or a callable. Check the scikit documentation of sklearn.svm.SVR for more details.
      function.

    OUTPUT:

    Classifier -- a ESVMClassifier instance.
    """

    def overHardMargin(self, x):
        return (not self.c) or x < self.c

    def __init__(self, solution, sample, unlabeled_sample, c, d, tube_tolerance,l,r,
                 kernel=LinearKernel(),  estimation_tolerance=0.001,debug_mode=False,**kwargs):

        r"""See ``ESVMClassifier`` for full documentation.

        """
        self.solution = solution
        alpha, gamma, delta = solution
        #tolerance on the variance in the list of estimations of threshold and tube radius.

        Classifier.__init__(self)
        self.kernel = kernel
        self.tube_tolerance =tube_tolerance
        self.c = c
        self.d = d
        self.debug_mode=debug_mode
        num_patterns = len(sample)
        num_unlabeled_patterns = len(unlabeled_sample)
        check_svm_classification_sample(sample)
        check_svm_classification_unlabeled_sample(unlabeled_sample)
        self.dim = len(sample[0].pattern)
        if len(alpha) != num_patterns:
            raise ValueError('The supplied sample and multipliers vector do \
                                not have the same size')

        #these multiplication are used both in the computation of b and epsilon, therefore are saved as variable
        all_signed_alpha = [alpha[i] * sample[i].label
                            for i in range(len(alpha))]
        all_gamma_delta = [gamma[s] - delta[s] for s in range(num_unlabeled_patterns)]

        if(debug_mode):
            self.unlabeled_sample=unlabeled_sample

            self.labeled_sample=sample

        #left part of the threshold b
        threshold_l_t = [sample[i].label -
                         sum([all_signed_alpha[j] *
                              self.kernel.compute(sample[j].pattern,
                                                  sample[i].pattern) for j in range(num_patterns)])
                         for i in range(num_patterns)
                         if alpha[i] > 0 and self.overHardMargin(alpha[i])]
        #right part of threshold b
        threshold_r_t = [sum([all_gamma_delta[s] *
                              self.kernel.compute(unlabeled_sample[s],
                                                  sample[i].pattern) for s in range(num_unlabeled_patterns)]) for i in
                         range(num_patterns)
                         if alpha[i] > 0 and self.overHardMargin(alpha[i])]

        threshold_list = [threshold_l_t[i] - threshold_r_t[i] for i in range(len(threshold_l_t))]

        if len(threshold_list) > 0:
            if max(threshold_list) - min(threshold_list) > math.fabs(estimation_tolerance * mean(threshold_list)):
                raise Exception("Variance on the estimation of the threshold is too high\n"
                                "Try using different parameters and a different kernel")

            self.threshold = mean(threshold_list)
        else:
            raise Exception("No valid samples to estimate the threshold")
        #indices of tube's support vectors

        gamma_tube_indices = [s for s in range(len(unlabeled_sample)) if gamma[s] > 0 and gamma[s] < d]
        delta_tube_indices = [s for s in range(len(unlabeled_sample)) if delta[s] > 0 and delta[s] < d]

        self.unlabeled_support_vectors_indices = [s for s in range(len(unlabeled_sample)) if
                                                  gamma[s] == 0 or delta[s] == 0]
        self.unlabeled_support_vectors = [unlabeled_sample[s] for s in self.unlabeled_support_vectors_indices]

        #indices of support vectors
        self.support_vectors_indices = [i for i in range(len(alpha)) if alpha[i] != 0]

        self.support_vectors = [sample[i].pattern for i in self.support_vectors_indices]
        self.signed_alphas = [alpha[i] * sample[i].label
                              for i in self.support_vectors_indices]
        self.gamma_delta = [gamma[s] - delta[s] for s in self.unlabeled_support_vectors_indices]

        """
        print [sum([all_signed_alpha[i]*self.kernel.compute(sample[i].pattern,unlabeled_sample[s])
                    for i in range(num_patterns)])+
                sum([all_gamma_delta[t]*self.kernel.compute(unlabeled_sample[t],unlabeled_sample[s])
                    for t in range(len(unlabeled_sample))])
                +self.threshold
                    for s in delta_tube_indices]
        """

        #print sum([all_signed_alpha[i]*self.kernel.compute(sample[i].pattern,unlabeled_sample[ind])
        #             for i in range(num_patterns)])
        #print sum([all_gamma_delta[t]*self.kernel.compute(unlabeled_sample[t],unlabeled_sample[ind])
        #            for t in range(len(unlabeled_sample))])
        #print self.threshold

        #print all_signed_alpha[0]*self.kernel.compute(sample[0].pattern,unlabeled_sample[ind])- all_gamma_delta[0]*self.kernel.compute(unlabeled_sample[0],unlabeled_sample[0])
        #+ self.threshold


        if len(delta_tube_indices) > 0:
            tube_radius_n = [(sum([all_signed_alpha[i] * self.kernel.compute(sample[i].pattern, unlabeled_sample[s])
                                  for i in range(num_patterns)]) +
                             sum([all_gamma_delta[t] * self.kernel.compute(unlabeled_sample[t], unlabeled_sample[s])
                                  for t in range(len(unlabeled_sample))])
                             + self.threshold)/r[s]
                             for s in delta_tube_indices]


        else:
            tube_radius_n = []

        if len(gamma_tube_indices) > 0:
            tube_radius_p = [(-sum([all_signed_alpha[i] * self.kernel.compute(sample[i].pattern, unlabeled_sample[s])
                                   for i in range(num_patterns)])-
                             sum([all_gamma_delta[t] * self.kernel.compute(unlabeled_sample[t], unlabeled_sample[s])
                                  for t in range(len(unlabeled_sample))])
                             - self.threshold)/l[s]
                             for s in gamma_tube_indices]
        else:
            tube_radius_p = []
        tube_list = tube_radius_n + tube_radius_p

        print tube_list
        if len(tube_list) > 0:

            if max(tube_list) - min(tube_list) > estimation_tolerance * math.fabs(mean(tube_list)):
                raise Exception("Variance on the estimation of the tube radius is too high\n"
                                "Try using different parameters and a different kernel")
            self.tube_radius = mean(tube_list)
        else:
            raise Exception("No valid samples to estimate the tube radius")

        if self.tube_radius > 0:
            self.in_tube_unlabeled_indices = [i for i in range(len(unlabeled_sample)) if gamma[i] < d and delta[i] < d]
        else:
            self.in_tube_unlabeled_indices = []

        """
        #regression

        regr=sklearn.svm.SVR(kernel="precomputed")
        x,y = [[x[:-1] for x in unlabeled_sample], [x[len(x)-2:][0] for x in unlabeled_sample]]

        gram=[[kernel.compute(i,j) for i in x] for j in x]

        regr.fit(gram,y)
        self.regr = regr

        alpha=self.signed_alphas
        beta=list(self.regr.dual_coef_)[0]
        x_i=self.support_vectors
        x_s=self.unlabeled_support_vectors
        r_i=self.regr.support_
        x_j=[x_s[s][:-1]+[0] for s in self.regr.support_]
        k=kernel.compute
        g_d=self.gamma_delta

        svmtr_l=sum([alpha[i]*beta[s]*k(x_i[i], x_j[s]) for s in range(len(x_j)) for i in range(len(x_i))])
        svmtr_r=sum([g_d[s]*beta[t]*k(x_j[t],x_s[s]) for t in range(len(r_i)) for s in range(len(x_s))])
        svmtr=svmtr_l+svmtr_r
        print svmtr
        norm_svm_l=sum([alpha[i]*alpha[j]*k(x_i[i],x_i[j]) for i in range(len(x_i)) for j in range(len(x_i))])
        norm_svm_c=sum([g_d[s]*g_d[t]*k(x_s[s],x_s[t]) for s in range(len(x_s)) for t in range(len(x_s))])
        norm_svm_r=2*sum([alpha[i]*g_d[t]*k(x_i[i],x_s[t]) for i in range(len(x_i)) for t in range(len(x_s))])
        norm_svm=math.sqrt(norm_svm_l+norm_svm_c+norm_svm_r)
        print norm_svm


        norm_r=math.sqrt(sum([beta[s]*beta[t]*k(x_j[s],x_j[t]) for s in range(len(x_j)) for t in range(len(x_j))]))

        print norm_r
        self.angle = math.degrees(math.acos(math.fabs(svmtr) / (norm_r * norm_svm)))
        print "angolo",self.angle

        m=MonteCarloSimulator()
        fa=lambda x:clf.decision_function(x)
        fb=lambda x:self.compute(x)
        print m.OverlappingArea(fa,fb,-3,3)
        """

    def regrPredict(self,X):
        if not self.debug_mode:
            raise Exception("This method is available only in debug mode")
        gram=[self.kernel.compute(p[:-1],X) for p in self.unlabeled_sample]
        return self.regr.predict(gram)

    def decision_function(self, pattern):
        if len(pattern) != self.dim:
            raise ValueError('The supplied pattern is incompatible with the \
    SVM dimension')

        kernel_values = [self.kernel.compute(x, pattern)
                         for x in self.support_vectors]
        kernel_unlabeled_values = [self.kernel.compute(x, pattern)
                                   for x in self.unlabeled_support_vectors]

        return dot(kernel_values, self.signed_alphas) + dot(kernel_unlabeled_values, self.gamma_delta) + self.threshold


    def intube(self, pattern):
        r"""
        INPUT:

        - ``self`` -- SVMClassifier object on which the function is invoked.

        - ``pattern`` -- pattern whose corresponding class is to be computed.

        OUTPUT: True if the supplied pattern is inside the epsilon-tube, False otherwise

        """
        distance = self.decision_function(pattern)
        return math.fabs(distance) < self.tube_radius or math.fabs(math.fabs(distance) - self.tube_radius) < self.tube_tolerance


    def compute(self, pattern):
        return sign(self.decision_function(pattern))