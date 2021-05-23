import json
import random


def sample_params(param_dist):
    """
    param_dist looks like eg:
    {
        'batch_size': [2,4,8,16],
        'vocab_size': [2,4,8,16]
    }
    """
    params = {}
    for k, v in param_dist.items():
        i = random.randint(0, len(v) - 1)
        sample = v[i]
        params[k] = sample
    return params


class RandomGridSearch(object):
    """
    things that might be nice to change:
    - store eg top 100, rather than just top 3
    - output everything that ever gets into top 100 to the logfile
    - still print top 3 to stdout each time
    - have a script to take the logfile, and print eg top 10, etc

    woudl also be good to count the number of total states
    and might be good to visit states without replacement
    """
    def __init__(self, train_fn, maximize, results_key, params_dist, logfile, print_params=False):
        self.train_fn = train_fn
        self.maximize = maximize
        self.results_key = results_key
        self.params_dist = params_dist
        self.logfile = logfile
        self.print_params = print_params

    def run(self):
        results_l = []
        max_results = 3
        it = 0
        f = open(self.logfile, 'w')
        while True:
            params = sample_params(self.params_dist)
            if self.print_params:
                print(params)
            results = self.train_fn(**params)
            score = results[self.results_key]
            if not self.maximize:
                score = -score
            if len(results_l) < max_results or score > results_l[-1]['score']:
                results_l.append({'params': params, 'score': score, 'results': results})
                results_l.sort(key=lambda x: x['score'], reverse=True)
                results_l = results_l[:3]
                print('it', it)
                for result in results_l:
                    print(result)
                if score == results_l[0]['score']:
                    logdict = params.copy()
                    logdict['score'] = score
                    logdict['it'] = it
                    logdict['results'] = results
                    f.write(json.dumps(logdict) + '\n')
                    f.flush()
            it += 1
