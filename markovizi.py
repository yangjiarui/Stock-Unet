




def start_optimize_Main():
    import scipy.optimize as spo

    def statistics(weights):
        global new_big_cap,new_small_cap,reit
        weights=np.array(weights)
        potfilo_return=np.sum(new_small_cap.mean()*weights)*252
        potfilo_std=np.sqrt(np.dot(weights.T,np.dot(new_small_cap.cov()*252,weights)))
        return np.array([potfileo_reture,potfilo_std,(potfilo_return-reit.mean())/potfilo_std])

    def min_sharpe(weights):
        return -statistics(weights)[2]
    def min_varians(weights):
        return statistics(weights)[1]
    cons=({'type':'eq','fun':lambda x:sum(x)-1})
    bnds=tuple((0,1) for x in range(len(new_small_cap.columns)))
    opts=spo.minimize(min_sharpe,len(new_small_cap.columns)*[1./len(new_small_cap.columns),],method='SLSQP',bounds=bnds,constraints=cons)

    # target_returns = np.linspace(0.0, 0.5, 50)
    #
    # target_variance = []
    #
    # for tar in target_returns:
    #     cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0]-tar}, {'type': 'eq', 'fun': lambda x: np.sum(x)-1})
    #
    #     res = spo.minimize(min_varians,len(new_small_cap.columns)*[1./len(new_small_cap.columns),], method='SLSQP', bounds=bnds, constraints=cons)
    #
    #     target_variance.append(res['fun'])

    #target_variance = np.array(target_variance)
  #  print(opts)

    #eturn #opts,
    return opts



q=start_optimize_Main()

