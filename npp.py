import numpy as np

def incoherence(x, sigma=None, cycle=False, dtype=np.float64):
    '''Returns how incoherent an array is with itself, in the order given. The incoherence is
        given with the root-mean-square method.
    Input:
        x: array-like:
            the array to measure the incoherence of.
        sigma: optional None or array-like, default None:
            the uncertainty of the x array. If this is included, it must have compatible 
            dimensions with x, and the output unit is then in sigmas.
        cycle: optional Bool, default False:
            whether the data is assumed to be cyclic and we expect x[0] and x[-1] to be similar.
        dtype: optional, default np.float64:
            the data type to use in calculations.
    Output:
        incoherence_value: float:
            on average, how close a point is to its neighbours. If sigma is included, the output
            unit is in sigmas, otherwise it is in the same unit as x.'''

    x = np.asarray(x, dtype=dtype)
    n = x.shape[0]

    incoherence_value = x[:-1] - x[1:]
    incoherence_value **= 2

    if sigma is not None:
        sigma = np.asarray(sigma, dtype=dtype)

        sigma_value = sigma[:-1]**2 + sigma[1:]**2
        incoherence_value /= sigma_value

    incoherence_value = np.sum(incoherence_value, axis=0)

    if cycle:
        edge = x[0] - x[-1]
        edge **= 2

        if sigma is not None:
            sigma_edge = sigma[0]**2 + sigma[-1]**2
            edge /= sigma_edge

        incoherence_value += edge
        incoherence_value /= n

    else:
        incoherence_value /= n - 1

    incoherence_value = np.sqrt(incoherence_value)
    incoherence_value /= 2

    return incoherence_value


def npp(P, t, x, sigma=None, dtype=np.float64):
    '''Returns the incoherence for a dataset with for a given period, or set of periods.
    Input:
        P: float or array-like:
            the Period(s) to measure the incoherence for.
        t: array-like:
            the time at wich the x measurements were taken.
        x: array-like,  must have compatible dimensions with t:
            the value of the x measurements
        sigma: optional None or array-like, default None:
            the uncertainty of the x array. If this is included, it must have compatible 
            dimensions with x, and the output unit is then in sigmas.
        dtype: optional, default np.float64:
            the data type to use in calculations.        
    Output:
        incoherence_value(s): float or array-like with the same shape as P:
            the incoherence value for each given period.'''

    P = np.asarray(P, dtype=dtype)
    t = np.asarray(t, dtype=dtype)
    x = np.asarray(x, dtype=dtype)

    if P.shape == ():
        t_phased = t % P
        sort_along = np.argsort(t_phased)

        x_sorted = x[sort_along]

        if sigma is None:
            return incoherence(x_sorted, cycle=True, dtype=dtype)

        else:
            sigma = np.asarray(sigma, dtype=dtype)
            sigma_sorted = sigma[sort_along]

            return incoherence(x_sorted, sigma_sorted, cycle=True, dtype=dtype)

    else:
        incoherence_values = np.empty(shape=P.shape)

        if sigma is None:
            for i, p in np.ndenumerate(P):
                t_phased = t % p
                sort_along = np.argsort(t_phased)

                x_sorted = x[sort_along]

                incoherence_values[i] = incoherence(
                    x_sorted, cycle=True, dtype=dtype)

        else:
            sigma = np.asarray(sigma, dtype=dtype)

            for i, p in np.ndenumerate(P):
                t_phased = t % p
                sort_along = np.argsort(t_phased)

                x_sorted = x[sort_along]
                sigma_sorted = sigma[sort_along]

                incoherence_values[i] = incoherence(
                    x_sorted, sigma_sorted, cycle=True, dtype=dtype)

        return incoherence_values


def auto_npp(t, x, P_min, P_max, density=0.1, sigma=None, dtype=np.float64):
    '''Returns test periods and incoherence values for a given dataset using a 
        neigbouring-point-periodogram.
    Input:
        t: array-like:
            the time at wich the x measurements were taken.
        x: array-like,  must have compatible dimensions with t:
            the value of the x measurements
        P_min: float, same unit as t:
            the minimum period to investigate, same unit as t
        P_max: float, same unit as t:
            the maximum period to investigate, same unit as t
        density: optional float, default 0.1:
            how tightly to space the test Periods. The tigther they are spaced, the smaller the
            periodic features that the periodogram can be expected to detect. A density of 0.1
            means that the periodogram can detect features lasting aproximately 10% of the Period.
        sigma: optional None or array-like, default None:
            the uncertainty of the x array. If this is included, it must have compatible 
            dimensions with x, and the output unit is then in sigmas.
        dtype: optional, default np.float64:
            the data type to use in calculations.        
    Output:
        (P_test, incoherence_values):
            P_test: array-like:
                the Periods tested by the periodogram, same unit as t
            incoherence_values: array-like, same shape as  P_test:
                the npp periodogram incoherence values for the test periods.'''

    t = np.asarray(t, dtype=dtype)
    x = np.asarray(x, dtype=dtype)

    t_width = np.max(t) - np.min(t)

    P = P_min
    P_test = [P]

    while(P < P_max):
        n = t_width / P
        P *= (n + density)/ n
        P_test.append(P)
    
    P_test.append(P_max)
    P_test = np.asarray(P_test)

    incoherence_values = npp(P_test, t, x, sigma, dtype)

    return P_test, incoherence_values