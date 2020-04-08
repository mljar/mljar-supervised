import os
import unittest
from supervised.utils.learning_curves import LearningCurves

class LearningCurvesTest(unittest.TestCase):
    def test_plot_close(self):
        '''
        Test if we close plots. To avoid following warning:
        RuntimeWarning: More than 20 figures have been opened. 
        Figures created through the pyplot interface (`matplotlib.pyplot.figure`) 
        are retained until explicitly closed and may consume too much memory.
        '''
        for _ in range(1): # you can increase the range, for tests speed reason I keep it low
            LearningCurves.plot_for_ensemble([3,2,1], "random_metrics", ".")

        os.remove(LearningCurves.output_file_name)
        