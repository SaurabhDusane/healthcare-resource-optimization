"""Statistical Testing Module"""

import pandas as pd
import numpy as np
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatisticalTester:
    """Perform statistical hypothesis tests."""
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.logger = logger
    
    def t_test(self, group1, group2, paired=False):
        """Perform t-test."""
        if paired:
            statistic, pvalue = stats.ttest_rel(group1, group2)
        else:
            statistic, pvalue = stats.ttest_ind(group1, group2)
        
        return {
            'statistic': statistic,
            'p_value': pvalue,
            'significant': pvalue < self.alpha,
            'effect_size': self._cohens_d(group1, group2)
        }
    
    def _cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def chi_square_test(self, observed, expected=None):
        """Perform chi-square test."""
        if expected is None:
            statistic, pvalue, dof, expected = stats.chi2_contingency(observed)
        else:
            statistic, pvalue = stats.chisquare(observed, expected)
            dof = len(observed) - 1
        
        return {
            'statistic': statistic,
            'p_value': pvalue,
            'dof': dof,
            'significant': pvalue < self.alpha
        }
    
    def anova_test(self, *groups):
        """Perform one-way ANOVA."""
        statistic, pvalue = stats.f_oneway(*groups)
        
        return {
            'statistic': statistic,
            'p_value': pvalue,
            'significant': pvalue < self.alpha
        }
    
    def correlation_test(self, x, y, method='pearson'):
        """Perform correlation test."""
        if method == 'pearson':
            corr, pvalue = stats.pearsonr(x, y)
        elif method == 'spearman':
            corr, pvalue = stats.spearmanr(x, y)
        else:
            corr, pvalue = stats.kendalltau(x, y)
        
        return {
            'correlation': corr,
            'p_value': pvalue,
            'significant': pvalue < self.alpha,
            'method': method
        }
    
    def normality_test(self, data):
        """Test for normality using Shapiro-Wilk."""
        statistic, pvalue = stats.shapiro(data)
        
        return {
            'statistic': statistic,
            'p_value': pvalue,
            'is_normal': pvalue > self.alpha
        }
