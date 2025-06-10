import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from optbinning import OptimalBinning
from pandas.api.types import is_numeric_dtype


def compute_woe_iv(predictor_df, feature, target_series, ordered=False, regulate=False):
    """
    Compute WoE and IV for a categorical feature.

    Parameters:
        predictor_df (pd.DataFrame): DataFrame containing the feature.
        feature (str): Name of the categorical feature column.
        target_series (pd.Series or pd.DataFrame): Series with binary target variable (1=good, 0=bad).
        ordered (bool): If False, sort result by WoE.
        regulate (bool): If True, adds small epsilon to avoid division/log(0).
    
    Returns:
        pd.DataFrame: WoE/IV table with total IV as a column.
    """
    eps = 1e-6 if regulate else 0

    # Combine into one DataFrame
    df = pd.concat([predictor_df[feature], target_series], axis=1)
    target_col = target_series.name if isinstance(target_series, pd.DataFrame) else target_series.name or 'target'
    df.columns = [feature, target_col]

    # Group by category
    grouped = df.groupby(feature, observed=True)[target_col].agg(['count', 'mean']).rename(
        columns={'count': 'cat_count', 'mean': 'prop_good'}
    )

    grouped['good'] = grouped['cat_count'] * grouped['prop_good']
    grouped['bad'] = grouped['cat_count'] * (1 - grouped['prop_good'])

    total_good = grouped['good'].sum()
    total_bad = grouped['bad'].sum()

    grouped['prop_good'] = grouped['good'] / total_good
    grouped['prop_bad'] = grouped['bad'] / total_bad

    grouped['WoE'] = np.log((grouped['prop_good'] + eps) / (grouped['prop_bad'] + eps))
    grouped['IV'] = (grouped['prop_good'] - grouped['prop_bad']) * grouped['WoE']
    grouped['%_of_obs'] = (grouped['cat_count'] / grouped['cat_count'].sum()) * 100
    grouped['total_IV'] = grouped['IV'].sum()

    if not ordered:
        grouped = grouped.sort_values(by='WoE')
        
    return grouped[['cat_count', '%_of_obs', 'good', 'bad', 'WoE', 'IV', 'total_IV']]





def plot_woe_with_bins(woe_df, xlabel=None, rot_of_xlabels=0, show=True):
    """
    Dual-axis plot showing % of observations and WoE by category.

    Parameters:
        woe_df (pd.DataFrame): Output of compute_woe_iv
        xlabel (str): Optional x-axis label
        rot_of_xlabels (int): Rotation angle for x-axis labels
        show (bool): Whether to call plt.show() inside the function
    """
    categories = woe_df.index.astype(str).tolist()
    woe_vals = woe_df['WoE'].values
    pct_obs = woe_df['%_of_obs'].values

    fig, ax1 = plt.subplots(figsize=(9, 4))

    # Bar chart for % of observations
    ax1.bar(categories, pct_obs, color='lightblue', alpha=0.8, label='% of Obs')
    ax1.set_xlabel(xlabel or woe_df.index.name or 'Category')
    ax1.set_ylabel('% of Observations')
    ax1.tick_params(axis='y')
    
    # Add transparent grid
    #ax1.grid(True, which='major', axis='y', linestyle='dotted', alpha=0.3)
    
    
    # x-axis label formatting
    ax1.set_xticks(np.arange(len(categories)))
    ax1.set_xticklabels(categories)
    plt.setp(ax1.get_xticklabels(), rotation=rot_of_xlabels)

    # Secondary y-axis for WoE
    ax2 = ax1.twinx()
    ax2.plot(range(len(categories)), woe_vals, color='red', marker='o', alpha = 0.5, linestyle='--', label='WoE')
    ax2.axhline(0, color='gray', linestyle='dotted', linewidth=1)
    ax2.set_ylabel('Weight of Evidence (WoE)', color='black')
    ax2.tick_params(axis='y')
    

    plt.title(f'WoE & % of Observations by "{xlabel or woe_df.index.name}"')
    plt.grid(alpha = 0.3)
    fig.tight_layout()

    if show:
        plt.show()
        
        
        

def get_woe_iv_optb(predictor_df, cat_var_name, target_var_df, optimize_bins = True, plot = False, **kwargs):
    
    """
    Compute Weight of Evidence (WoE) and Information Value (IV) for a categorical variable,
    optionally using optimal binning via the optbinning library.
    
    Parameters:
    predictor_df (pd.DataFrame): DataFrame containing the predictor variables.
    cat_var_name (str): Name of the categorical variable to bin.
    target_var_df (pd.Series or pd.DataFrame): Binary target variable (0/1).
    optimize_bins (bool): If True, use optbinning to optimize binning. If False,
                          use original categories as bins.
    plot (bool): If True, show WoE plot after binning.
    **kwargs: Additional arguments passed to `OptimalBinning`.
    
    Returns:
    woe_summary_table (pd.DataFrame): Table containing bin stats, including WoE and IV.
    binning_table (BinningTable): Full binning table object from optbinning.
    optb (OptimalBinning): Fitted binning object.
    """
    
    # Ensure target is a Series, even if passed as a DataFrame
    target_series = target_var_df.squeeze()
    predictor_series = predictor_df[cat_var_name]
    
    # Validate input data
    if cat_var_name not in predictor_df.columns:
        
        raise ValueError(f"{cat_var_name} not found in predictor_df columns.")
    
    if not set(pd.Series(target_series).astype(int).unique()).issubset({0, 1}):
        raise ValueError("Target variable must be binary (0/1 or True/False).")

    
    is_numeric = is_numeric_dtype(predictor_series)
    bin_dtype = "numerical" if is_numeric else "categorical"
    
    # Build base kwargs for OptimalBinning
    binning_kwargs = {
        "name": cat_var_name,
        "dtype": bin_dtype,
        "solver": "cp",
        "divergence": "iv",
        **kwargs
    }
    
    # Add monotonic trend only for numerical variables
    if is_numeric and "monotonic_trend" not in kwargs:
        
        binning_kwargs["monotonic_trend"] = "auto"
        
    if not optimize_bins:
        
            binning_kwargs["dtype"] = "categorical"
            binning_kwargs["user_splits"] = [[cat] for cat in predictor_series.unique()]
        #else:
            #raise ValueError("optimize_bins=False is only supported for categorical variables.")
    
    
    # Initialize Optbinning and fit 
    optb = OptimalBinning(**binning_kwargs)
    optb.fit(predictor_series.values, target_series.values)
    
    # Get a summary table including WoE, IV, etc.
    binning_table = optb.binning_table
    woe_summary = binning_table.build()
    
    # Adjust the sign conventions of optbinning in computing WoE
    woe_summary["WoE"] = -1 * woe_summary["WoE"]
    
    # Identify reference bin (lowest WoE)
    ref_woe = pd.to_numeric(woe_summary.loc[:,'WoE']).min()
    
    # add a boolean column to indicate the reference bin
    woe_summary.loc[:,'Ref_Bin'] = (woe_summary['WoE'] == ref_woe)
    
    if plot:
        
        optb.binning_table.plot(metric="woe")
        
    return woe_summary, binning_table, optb



def variable_bins_diagnose(predictor_df, cat_vars, target_df, **kwargs):
    
    """
    Diagnose raw binning vs optimized binning for categorical variables.

    Parameters:
        predictor_df : DataFrame of predictors
        cat_vars     : List of categorical variable names
        target_df    : Series or DataFrame with target labels
        **kwargs     : Additional arguments passed to get_woe_iv_optb

    Returns:
        pd.DataFrame with IVs, quality scores, and reference bins
    """
    
    frames = []

    for cat_var in cat_vars:
        
        woe_t, bin_table, _ = get_woe_iv_optb(predictor_df, cat_var, target_df, optimize_bins=False)
        opt_woe_t, opt_bin_table, _ = get_woe_iv_optb(predictor_df, cat_var, target_df, **kwargs)

        bin_table.analysis(print_output=0)
        opt_bin_table.analysis(print_output=0)

        ref_bin = woe_t['Bin'][woe_t['Ref_Bin']].values[0]
        ref_bin_opt = opt_woe_t['Bin'][opt_woe_t['Ref_Bin']].values[0]

        row = pd.DataFrame([{
            'Variable': cat_var,
            'IV': bin_table.iv,
            'Opt_IV': opt_bin_table.iv,
            'QS': f"{bin_table.quality_score:.5f}",
            'Opt_QS': f"{opt_bin_table.quality_score:.5f}",
            'Ref_Bin': ref_bin,
            'Opt_Ref_Bin': ref_bin_opt
        }])
        frames.append(row)

    return pd.concat(frames, ignore_index=True)



def get_dummies_custom(predictor_df, cat_var_name, optb):
    
    """
    Generate dummies 

    Inputs:
        predictor_df : DataFrame of predictors
        cat_var_name  : Categorical variable name
        optb: fitted optbinning object

    Returns:
        pd.DataFrame for the dummified variable
    """
    
    # get bin labels for each instance in the predictor df
    bin_labels = optb.transform(predictor_df[cat_var_name].values, metric="bins")
    
    # turn them into a series
    grade_bin_series = pd.Series(bin_labels)
    
    # clean labels 
    clean_labels = grade_bin_series.str.replace(r"[\[\]']", "", regex=True).str.replace(" ", "-")

    dummies_df = pd.get_dummies(clean_labels, prefix = cat_var_name, prefix_sep=':')
    
    return dummies_df