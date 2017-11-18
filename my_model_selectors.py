import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_num_components, min_BIC_score = None, None
        for num_components in range(self.min_n_components, self.max_n_components+1):
        	try:
        		logL = self.base_model(num_components).score(self.X, self.lengths)
        		logN = np.log(len(self.X))
        		p = num_components ** 2 + 2 * len(self.X[0]) * num_components - 1
        		BIC_score = -2 * logL + p * logN
        		if min_BIC_score is None or min_BIC_score > BIC_score:
        			best_num_components, min_BIC_score = num_components, BIC_score
        	except:
        		pass
        return self.base_model(best_num_components)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_num_components, max_DIC_score = None, None
        for num_components in range(self.min_n_components, self.max_n_components+1):
        	try:
        		model = self.base_model(num_components)
        		logPXi = model.score(self.X, self.lengths)
        		sum_log = 0.
        		count_log = 0
        		for word in self.hwords.keys():
        			if word != self.this_word:
        				X_other, lengths_other = self.hwords[word]
        				sum_log += model.score(X_other, lengths_other)
        				count_log += 1
        		DIC_score = logPXi - sum_log / count_log
        		if max_DIC_score is None or max_DIC_score < DIC_score:
        			best_num_components, max_DIC_score = num_components, DIC_score
        	except:
        		pass
        return self.base_model(best_num_components)



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_num_components, max_avg_LogL = None, None
        for num_components in range(self.min_n_components, self.max_n_components+1):
        	sum_LogL, count_LogL = 0., 0
        	try:
        		split_method = KFold(min(len(self.lengths), 3))
        		for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
        			X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
        			X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
        			try:
        				cv_model = self.base_model(num_components).fit(X_train, lengths_train)
        				sum_LogL += cv_model.score(X_test, lengths_test)
        				count_LogL += 1
        			except:
        				pass
        		avg_LogL = sum_LogL / count_LogL
        		if max_avg_LogL is None or max_avg_LogL < avg_LogL:
        			max_avg_LogL, best_num_components = avg_LogL, num_components
        	except:
        		pass
        return self.base_model(best_num_components)