"""

Read a checkpoints and export its weights to a numpy variable

"""

# import os
import sys

import numpy as np
import tensorflow as tf


def getTensor(file_name, weight_name):
    try:
        reader = tf.train.NewCheckpointReader(file_name)
        print(reader.debug_string().decode("utf-8"))
        if(weight_name):
            # order is [filter_height, filter_width, input_depth, output_depth]
            weight_array = reader.get_tensor(weight_name)
            print(weight_array.shape)
            print(weight_array[3][4][5][45])
            print(weight_array.transpose().shape)
            print(weight_array.transpose()[45][5][4][3])

    except Exception as e:  # pylint: disable=broad-except
        print(str(e))


class DnnWeights:
    @staticmethod
    def readFromFile(file_name, weight_name):
        reader = tf.train.NewCheckpointReader(file_name)
        weights = reader.get_tensor(weight_name).transpose()
        return DnnWeights(weight_name, weights)

    def __init__(self, name, weights):
        self.name = name
        self.weights = weights
        self.weights_flat = weights.flatten()
        self.shape = weights.shape

    def _buildPrunningTable(self, thFun, thParList):
        output = None
        # print("starting pruning")
        for weightVector in self.weights:
            threshold = thFun(weightVector, thParList)
            # print("threshold: %f" % (threshold))
            flagArray = np.array([np.absolute(weightVector) < threshold])
            output = (flagArray if output is None
                      else np.append(output, flagArray, axis=0))
        return output

    def getPrunningTable(self, method,
                         staticTh=None, cuttoff=None, pctMax=None):

        def dynThFun(weightVector, parList):
            assert len(parList) == 2
            pctMax = parList[0]
            threshold = parList[1]
            return min(np.max(np.absolute(weightVector.flatten()))*pctMax,
                       threshold)

        def dyn2ThFun(weightVector, parList):
            assert len(parList) == 1
            cuttoff = parList[0]
            """print("cuttoff: %f cuttoff idx: %f total elem: %d" %
                  (cuttoff,
                   weightVector.size * cuttoff,
                   weightVector.size))"""
            return np.sort(np.absolute(
                weightVector.flatten()))[-int(weightVector.size *
                                              cuttoff)]

        def staticThFun(weightVector, parList):
            assert len(parList) == 1
            return parList[0]

        if method == "static":
            thFun = staticThFun
            parList = [staticTh]
        elif method == "dyn1":
            thFun = dynThFun
            parList = [pctMax, staticTh]
        elif method == "dyn2":
            thFun = dyn2ThFun
            parList = [cuttoff]
        else:
            raise NameError("error unknown prunning method")

        return self._buildPrunningTable(thFun, parList)

    def countPrunnedElements(self, method,
                             staticTh=None, cuttoff=None, pctMax=None):
        return np.add.reduce(self.getPrunningTable(method, staticTh,
                                                   cuttoff, pctMax).flatten())

    def compareLayers(self, layer, threshold):
        """ Compare two layers, find how many low value weights both have,
            how many low value weighs one have but the other does not,
            and the other way around """
        lowWeightsL1 = self.lowWeights(threshold)
        lowWeightsL2 = layer.lowWeights(threshold)
        zeroInBoth = lowWeightsL1 & lowWeightsL2
        zeroOnlyInFirst = lowWeightsL1 & ~lowWeightsL2
        zeroOnlyInSecond = lowWeightsL2 & ~lowWeightsL1

        return (zeroOnlyInFirst, zeroOnlyInSecond, zeroInBoth)


class WeightExtract:
    def __init__(self, file_name, weight_names):
        self.file_name = file_name
        self.weight_names = weight_names


def main():
    filename = sys.argv[1]
    if len(sys.argv) > 2:
        weight_name = sys.argv[2]
    else:
        weight_name = None
    getTensor(filename, weight_name)


def numpy_ufunc_test():
    X = np.arange(8)
    compResult = np.less(X, 3)
    reduc = np.add.reduce(compResult)

    print(X)
    print(compResult)
    print(reduc)

    print(np.add.reduce(np.less(X, 3)))


def testDnnWeights(base_file_name, step1, step2, weight_name):
    w1 = DnnWeights.readFromFile("%s-%d" % (base_file_name, step1),
                                 weight_name)
    w2 = DnnWeights.readFromFile("%s-%d" % (base_file_name, step2),
                                 weight_name)

    neuronTested = 2
    dynThreshold = 1/.256
    cuttoff1 = .95
    cuttoff2 = .40
    threshold1 = .001
    threshold2 = .1

    p_start = 0
    p_end = 1024
    print("neuron tested: %d, threshold1: %f, threshold2: %f" %
          (neuronTested, threshold1, threshold2))

    def printStats(name, layer, methods,
                   staticTh=None, cuttoff=None, pctMax=None):

        print(">>>> %s stats:" % (name))
        print(np.array_str(np.sort(layer.weights[neuronTested].flatten()[p_start:p_end]),
                           precision=3, suppress_small=True))
        for method in methods:
            print("## %s stats: " % (method))
            prunnedElem = layer.countPrunnedElements(method, staticTh,
                                                     cuttoff, pctMax)
            print("     zero elements: %d, %f %%"
                  % (prunnedElem,
                     100.0 * (prunnedElem /
                              len(layer.weights.flatten()))))

    printStats("w1", w1, ["static", "dyn1", "dyn2"],
               threshold1, cuttoff1, dynThreshold)
    printStats("w2", w2, ["static", "dyn1", "dyn2"],
               threshold2, cuttoff2, dynThreshold)

    zeroValues1 = w1.getPrunningTable("dyn2", cuttoff=cuttoff1)
    zeroValues2 = w2.getPrunningTable("dyn2", cuttoff=cuttoff2)

    zeroBoth = zeroValues1 & zeroValues2
    zeroL1 = zeroValues1 & ~zeroValues2
    zeroL2 = ~zeroValues1 & zeroValues2
    zeroSome = zeroValues1 | zeroValues2

    def printZeroArray(name, array):
        print("%s elements %d:" % (name, np.add.reduce(array.flatten())))
        # print(np.array_str(array[neuronTested][p_start:p_end]))

    printZeroArray("zero both", zeroBoth)
    printZeroArray("zeroL1", zeroL1)
    printZeroArray("zeroL2", zeroL2)
    printZeroArray("zeroSome", zeroSome)


np.set_printoptions(threshold=np.nan)
if __name__ == "__main__":
    testDnnWeights(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                   sys.argv[4])
