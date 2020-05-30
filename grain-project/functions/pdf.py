
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm
import matplotlib


matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')



# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """
    Model data by finding best fit distribution to data
    """
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [        
        st.beta,st.gamma,st.lognorm,
        st.norm,st.weibull_min,st.weibull_max
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        label = distribution.name
                        pd.Series(pdf, x).plot(ax=ax, label=label, legend=True)
                        ax.legend(loc='upper right')
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)

def make_pdf(dist, params, size=10000):
    """
    Generate distributions's Probability Distribution Function 
    """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf



def find_PDF(data_all, data_name):
    """
    それぞれのパラメータのProbability Density Functionを見つける
    Params
    ----------------------------------------------------
        data_all = [parameter1, parameter2, ...]
        data_name = [param_name1, param_name2, ...]
    
    Returns
    ----------------------------------------------------
        PDF
    
    example
    ----------------------------------------------------
    data_all = [Areas, Circularities, Eq_diameters, Shortest, Longest, Perimeters]
    data_name = ['Areas', 'Circularities', 'Eq_diameters', 'Shortest', 'Longest', 'Perimeters']
    """
    
    for i in range(len(data_all)):
        # Load data from statsmodels datasets
        data = pd.Series(data_all[i])
    
        # Find best fit distribution
        best_fit_name, best_fit_params = best_fit_distribution(data, 200, ax=None)
        best_dist = getattr(st, best_fit_name)

        # Make PDF with best params 
        pdf = make_pdf(best_dist, best_fit_params)

        param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
        dist_str = '{}({})'.format(best_fit_name, param_str)

        print(f'{str(data_name[i])}: {dist_str}')