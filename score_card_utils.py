import pandas as pd
import numpy as np

class CreditScorecard:
    
    def __init__(self, model_df: pd.DataFrame, base_score: float = 300, max_score: float = 850):
        """
        model_df: A DataFrame with columns ['Variable', 'Coefficient'], where
                  'Variable' is in the format 'variable_name:category'
        """
        self.model_df = model_df.copy()
        self.model_df.reset_index(inplace=True) #no drop = True

        # Extract original variable names
        self.model_df['original_variable'] = self.model_df.variable_name.str.split(':').str[0]

        # Save score range
        self.base_score = base_score
        self.max_score = max_score
        self.score_range = max_score - base_score

        # Compute min and max sum of coefficients for scaling later
        self.min_sum_of_coeff = (
            self.model_df.groupby('original_variable')['coefficient'].min().sum()
        )
        self.max_sum_of_coeff = (
            self.model_df.groupby('original_variable')['coefficient'].max().sum()
        )
        
    def generate_scorecard(self) -> pd.DataFrame:
        #To-Do: handle possible rounding issues
        """
        Scales coefficients into scorecard points and returns a scorecard DataFrame.
        Intercept ('const') is scaled separately and added to the first row.
        """
                
        score_card = self.model_df.copy()
        
        # Compute scaled scores
        score_card['score'] = (score_card['coefficient'].values * self.score_range
                               / (self.max_sum_of_coeff - self.min_sum_of_coeff))

        # Scale intercept ('const') separately
        intercept_idx = score_card.index[score_card['variable_name'] == 'const'].tolist()[0]
        intercept_coef = score_card.loc[intercept_idx, 'coefficient']
        
        score_card.loc[intercept_idx, 'score'] = ((intercept_coef - self.min_sum_of_coeff)
                                                  /(self.max_sum_of_coeff - self.min_sum_of_coeff)
                                                  * self.score_range
                                                  + self.base_score)
        # round the results 
        score_card['score'] = score_card['score'].round()
        
        score_card = score_card.drop('original_variable', axis = 1)

        return score_card
    
    def compute_credit_score(self, Xval_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate credit scores for dummified validation/test data using the scorecard.

        Parameters:
        - Xval_df: DataFrame of dummified features (no intercept column).

        Returns:
        - DataFrame with a single 'Score' column.
        """
        # Generate the scorecard
        score_card_df = self.generate_scorecard()
        
        # Copy validation data
        Xval = Xval_df.copy()

        # Add intercept (const) column
        Xval.insert(0, 'const', 1)

        # Ensure columns match the score_card variable order
        Xval = Xval[score_card_df.variable_name.values]

        # Get the score values
        scores = score_card_df.score.values.reshape(-1,1)

        # Compute the dot product
        val_scores = Xval.dot(scores)
        
        val_scores.columns = ['predicted_score']

        return val_scores
    
    def get_pd_from_score(self, score_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a DataFrame of credit scores into estimated probabilities of non-default (1 - PD).

        Parameters:
        - score_df: DataFrame with a single column of credit scores (e.g., from `calculate_credit_score()`)

        Returns:
        - DataFrame with a single column 'P_Good' representing estimated probability of good
        """
        # Convert scaled score to raw log-odds
        sum_of_coeffs = (((score_df - self.base_score) / self.score_range)
                         * (self.max_sum_of_coeff - self.min_sum_of_coeff)
                         + self.min_sum_of_coeff)
        
        # Step 2: Convert to NumPy array for safe exponentiation
        log_odds = sum_of_coeffs.to_numpy().astype(float)
        
        # Convert log-odds to probability of good
        prob_good = np.exp(log_odds) / (1 + np.exp(log_odds))
        
        prob_good_df = pd.DataFrame(prob_good)
        prob_good_df.columns =['yhat_proba']
        
        return prob_good_df
    
    def get_threshold_summary(self, prob_df: pd.DataFrame, thresholds, fpr, tpr) -> pd.DataFrame:
        """
        For each probability threshold, compute:
            - FPR, TPR
            - Corresponding score
            - Approval/rejection counts and rates

        Parameters:
            - prob_df: DataFrame with column 'yhat_proba' — model-predicted probabilities (of GOOD or BAD, depends on your convention)
            - thresholds: List/array of probability thresholds (e.g. from ROC)
            - fpr: False positive rates
            - tpr: True positive rates

        Returns:
            - DataFrame with: threshold, fpr, tpr, score, approved_count, rejected_count, approval_rate, rejection_rate
        """
        df = pd.DataFrame({'threshold': thresholds, 'fpr': fpr, 'tpr': tpr})

        # Ensure threshold[0] is 1.0
        df.loc[0, 'threshold'] = 1.0

        # Compute log-odds from prob threshold
        log_odds = np.log(df['threshold'] / (1 - df['threshold']))

        # Convert to score
        df['score'] = ((log_odds - self.min_sum_of_coeff)
                   * self.score_range / (self.max_sum_of_coeff - self.min_sum_of_coeff)
                   + self.base_score)

        # For each threshold, compute approval/rejection stats
        approval_counts = []
        rejection_counts = []

        for prob_cut in df['threshold']:
            
            is_approved = prob_df['yhat_proba'] >= prob_cut
            
            approved = is_approved.sum()
            rejected = len(prob_df) - approved
            
            approval_counts.append(approved)
            rejection_counts.append(rejected)

        total = len(prob_df)
        
        df['approved_count'] = approval_counts
        df['rejected_count'] = rejection_counts
        df['approval_rate'] = df['approved_count'] / total
        df['rejection_rate'] = df['rejected_count'] / total

        return df


