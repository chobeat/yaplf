r"""
Package handling variable-quality SV classification learning algorithms in
yaplf.

Package yaplf.algorithms.svm.classification.vq contains all the classes
handling variable-quality SV classification learning algorithms in yaplf.

- pep8 checked
- pylint score: 9.63

AUTHORS:

- Dario Malchiodi (2013-09-04): initial version.


"""

#*****************************************************************************
#       Copyright (C) 2013 Dario Malchiodi <malchiodi@di.unimi.it>
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


from yaplf.algorithms.svm.vq.solvers import CVXOPTVQClassificationSolver

class SVMVQClassificationAlgorithm(LearningAlgorithm):
    r"""
    SVM Classification Algorithm for data of variable quality, as described in
    [Apolloni et al., 2007].

    INPUT:

    - ``sample`` -- list or tuple of ``LabeledExample`` instances whose
      labels are all set either to `1` or `-1`.

    - ``c_value`` -- float (default: None, amounting to the hard-margin version
      of the algorithm) value for the trade-off constant `C` between steepness
      and accuracy in the soft-margin version of the algorithm.

    - ``kernel`` -- ``Kernel`` (default: ``LinearKernel()``) instance defining
      the kernel to be used.

    OUTPUT:

    ``LearningAlgorithm`` instance.

    EXAMPLES:

    Variable-quality SV classification algorithm can be directly applied to any
    problem whose single examples are explicitly associated to a numerical
    evaluation of their quality. For instance, consider the following extension
    of the binary XOR problem, whose upper-left example is given higher
    importance:

    ::

        >>> from yaplf.data import LabeledExample, AccuracyExample
        >>> xor_sample = [AccuracyExample(LabeledExample((1, 1), -1), 0),
        ... AccuracyExample(LabeledExample((0, 0), -1), 0),
        ... AccuracyExample(LabeledExample((0, 1), 1), .3),
        ... AccuracyExample(LabeledExample((1, 0), 1), 0)]

    Training a SV classifier on this sample using a polynomial kernel brings to
    the following model:

    ::

        >>> from yaplf.algorithms.svm.classification \
        ... import SVMVQClassificationAlgorithm
        >>> from yaplf.models.kernel import PolynomialKernel
        >>> alg = SVMVQClassificationAlgorithm(xor_sample,
        ... kernel = PolynomialKernel(4))
        >>> alg.run()
        >>> alg.model
        SVMClassifier([0.092013250246483894, 0.39872408400362103,
        0.25518341380777959, 0.23555392044232534], -1.1501656251,
        [LabeledExample((1, 1), -1.0), LabeledExample((0, 0), -1.0),
        LabeledExample((0, 1), 1.0), LabeledExample((1, 0), 1.0)],
        kernel = PolynomialKernel(4))

    The actual exploitation of the additional information about data quality
    can be tested plotting the SV classifier decision function and noting that
    the class boundary is farther from the upper-left example than from the
    other ones:

    ::

        >>> alg.model.plot((0, 1), (0, 1), shading = True)

    REFERENCES:

    [Apolloni et al., 2007] B. Apolloni, D. Malchiodi and L. Natali, A Modified
    SVM Classification Algorithm for Data of Variable Quality, in
    Knowledge-Based Intelligent Information and Engineering Systems 11th
    International Conference, KES 2007, XVII Italian Workshop on Neural
    Networks, Vietri sul Mare, Italy, September 12-14, 2007. Proceedings, Part
    III, Berlin Heidelberg: Springer-Verlag, Lecture Notes in Artificial
    Intelligence 4694 (ISBN 978-3-540-74828-1), 131-139, 2007

    AUTHORS:

    - Dario Malchiodi (2010-04-12)

    """

    def __init__(self, sample, **kwargs):
        r"""
        See ``SVMVQClassificationAlgorithm`` for full documentation.

        """

        red_sample = [a.example for a in sample]
        LearningAlgorithm.__init__(self, red_sample)
        check_svm_classification_sample(red_sample)
        self.sample = sample
        self.model = None

        try:
            self.c_value = kwargs['c_value']
        except KeyError:
            self.c_value = None

        try:
            self.kernel = kwargs['kernel']
        except KeyError:
            self.kernel = LinearKernel()

        self.solver = CVXOPTVQClassificationSolver()

    def run(self):
        r"""
        Run the variable-quality SVM classification learning algorithm.

        INPUT:

        No input.

        OUTPUT:

        No output. After the invocation the inferred model is available through
        the ``model`` field, in form of a ``SVMClassifier`` instance.

        EXAMPLES:

        Variable-quality SV classification algorithm can be directly applied to
        any problem whose single examples are explicitly associated to a
        numerical evaluation of their quality. For instance, consider the
        following extension of the binary XOR problem, whose upper-left example
        is given higher importance:

        ::

            >>> from yaplf.data import LabeledExample, AccuracyExample
            >>> xor_sample = [AccuracyExample(LabeledExample((1, 1), -1), 0),
            ... AccuracyExample(LabeledExample((0, 0), -1), 0),
            ... AccuracyExample(LabeledExample((0, 1), 1), .3),
            ... AccuracyExample(LabeledExample((1, 0), 1), 0)]

        Training a SV classifier on this sample using a polynomial kernel
        brings to the following model:

        ::

            >>> from yaplf.algorithms.svm.classification \
            ... import SVMVQClassificationAlgorithm
            >>> from yaplf.models.kernel import PolynomialKernel
            >>> alg = SVMVQClassificationAlgorithm(xor_sample,
            ... kernel = PolynomialKernel(4))
            >>> alg.run()
            >>> alg.model
            SVMClassifier([0.092013250246483894, 0.39872408400362103,
            0.25518341380777959, 0.23555392044232534], -1.1501656251,
            [LabeledExample((1, 1), -1.0), LabeledExample((0, 0), -1.0),
            LabeledExample((0, 1), 1.0), LabeledExample((1, 0), 1.0)],
            kernel = PolynomialKernel(4))

        The actual exploitation of the additional information about data
        quality can be tested plotting the SV classifier decision function and
        noting that the class boundary is farther from the upper-left example
        than from the other ones:

        ::

            >>> alg.model.plot((0, 1), (0, 1), shading = True)

        AUTHORS:

        - Dario Malchiodi (2010-04-12)

        """

        alpha = self.solver.solve(self.sample, self.c_value, self.kernel)
        num_examples = len(self.sample)

        if self.c_value == None:
            threshold = mean([self.sample[i].example.label -
                sum([alpha[j] * self.sample[j].example.label *
                self.kernel.compute(self.sample[j].example.pattern,
                self.sample[i].example.pattern)
                for j in range(num_examples)]) for i in range(num_examples)
                if alpha[i] > 0])
        else:
            threshold = mean([self.sample[i].example.label -
                sum([alpha[j] * self.sample[j].example.label *
                self.kernel.compute(self.sample[j].example.pattern,
                self.sample[i].example.pattern) for j in range(num_examples)])
                for i in range(num_examples)
                if alpha[i] > 0 and alpha[i] < self.c_value])

        self.model = SVMClassifier(alpha, threshold, [elem.example \
            for elem in self.sample], kernel=self.kernel)

