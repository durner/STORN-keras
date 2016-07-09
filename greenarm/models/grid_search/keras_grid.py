from sklearn.grid_search import GridSearchCV

from greenarm.util import get_logger
import logging

logger = get_logger("grid_search")
hdlr = logging.FileHandler('results/grid_search/grid.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)


class ModelSelector:
    def __init__(self, model):
        self.model = model

    def score_hyper_params(self, inputs, target, param_grid):
        """

        :param inputs:
        :param target:

        :param param_grid:
               param_grid = dict(optimizer=['rmsprop', 'adam'], nb_epoch=[150, 180, 200], batch_size=[32, 44, 100])

               params can contain any params from the fit() function of the model, and from the build function as well
        :return:
        """
        grid = GridSearchCV(estimator=self.model, param_grid=param_grid)
        result = grid.fit(inputs, target)

        # summarize
        logger.info("Best: %f using %s" % (result.best_score_, result.best_params_))
        for params, mean_score, scores in result.grid_scores_:
            logger.info("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

        return result
