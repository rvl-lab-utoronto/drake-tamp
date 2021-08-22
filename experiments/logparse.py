from glob import glob
import json
import os
import re
import numpy as np

def dict_from_string(s:str):
    obj = eval(s, type('js', (dict,), dict(__getitem__=lambda s, n: n))())
    return obj

def parse_logs(exp_dir, exp_name):
    data = []
    for i, log_filename in enumerate(glob(os.path.join(exp_dir, '*.log'))):
        with open(log_filename, 'r') as f:
            d = {
                'solved': False,
                'exp_name': exp_name,
                'run':os.path.basename(log_filename).strip('.log'),
                'planning_iters': 0,
                'planning_time': 0,
                'expanded': float('nan')
            }

            data.append(d)
            for line in f.readlines():
                if line.startswith('Planning'):
                    i_star, i_ground, q = re.search('Planning. # optim: (\d+). # grounded: (\d+). Queue length: (\d+)', line).groups()
                    d['results'] = int(i_star)
                    d['evaluations'] = int(i_ground)
                    d['queue'] = int(q)
                    d['planning_iters'] += 1

                    match = re.search('. Expanded: (\d+). duration: ([0-9.]+)$', line)
                    if match:
                        expanded, duration = match.groups()
                        d['expanded'] = int(expanded)
                        d['planning_time'] += float(duration)
                if 'Results' in line:
                    (results, ) = re.search('Results: (\d+)', line).groups()
                    d['results'] = max(int(results), d.get('results', 0))
                if 'Attempt' in line:
                    d['planning_iters'] += 1
                if line.startswith('Summary:'):
                    d.update(dict_from_string(line[8:].replace('inf', '"inf"')))
    return data

def load_results_from_stats(exp_dir, exp_name):
    data = []
    for i, stats_path in enumerate(glob(os.path.join(exp_dir, '*_logs/stats.json'))):
        with open(stats_path, 'r') as f:
            run_stats = json.load(f)
        d = run_stats['summary']
        d['results'] = max(run_stats['results'][1])
        d['evaluations'] = max(run_stats['evaluations'][1])
        # fd stats
        d['planning_iters'] = len(run_stats['fd_stats'])
        d['total_expanded'] = sum([p.get('expanded', 0) for p in run_stats['fd_stats']])
        d['median_expanded'] = np.median([p.get('expanded', 0) for p in run_stats['fd_stats']])
        d['mean_expanded'] = np.mean([p.get('expanded', 0) for p in run_stats['fd_stats']])
        d['max_expanded'] = max([p.get('expanded', 0) for p in run_stats['fd_stats']])
        d['total_evaluated'] = sum([p.get('evaluated', 0) for p in run_stats['fd_stats']])
        d['max_evaluated'] = max([p.get('evaluated', 0) for p in run_stats['fd_stats']])
        d['total_fd_search_time'] = sum([p.get('total_time', 0) for p in run_stats['fd_stats']])
        d['total_translation_time'] = sum([p.get('translation_time', 0) for p in run_stats['fd_stats']])
        d['total_fd_timeouts'] = sum([p.get('timeout', 0) for p in run_stats['fd_stats']])
        d['scoring_time'] = run_stats.get('scoring_time', float('nan'))
        d['exp_name'] = exp_name
        d['run'] = stats_path.split(os.path.sep)[-2].strip(".yaml_logs")
        data.append(d)
    return data

def compare_same_set(data, only_solved=False):
    runs = {}
    for d in data:
        if not only_solved or d.get('solved'):
            runs.setdefault(d['exp_name'], set()).add(d['run'])

    include = set()
    for exp in runs:
        include = include & runs[exp] if include else runs[exp]

    return [d for d in data if d['run'] in include]

def table_compare(data):
    data = compare_same_set(data)
    df = pd.DataFrame(data)

    df.loc[~df.solved, 'run_time'] = float('nan')
    df.loc[~df.solved, 'planning_time'] = float('nan')
    df.loc[~df.solved, 'sample_time'] = float('nan')
    df.loc[~df.solved, 'search_time'] = float('nan')
    df.loc[~df.solved, 'results'] = float('nan')
    df.loc[~df.solved, 'evaluations'] = float('nan')

    aggregates = df.groupby('exp_name').agg(['sum','mean', 'std'])[[
            ['solved', 'sum'],
            ['solved', 'mean'],
            ['run_time', 'mean'],
            ['run_time', 'std'],
            ['search_time', 'mean'],
            ['search_time', 'std'],
            ['sample_time', 'mean'],
            ['sample_time', 'std'],
            ['results', 'mean'],
            ['results', 'std'],
            ['evaluations', 'mean'],
            ['evaluations', 'std'],
        ]
    ]
    print(aggregates.to_string(float_format="%.2f"))
    # print(aggregates.to_latex(float_format="%.2f", bold_rows=True))

def num_blocks_vs_max_stack(data):
    for d in data:
        d['num_blocks'], d['num_blockers'], d['max_stack'], _ = map(int, d['run'].strip('.yaml').split('_'))

    df = pd.DataFrame(data)

    d = df.pivot_table(columns='max_stack', values='solved', index='num_blocks', aggfunc='mean')
    print(d.to_string(float_format="%.2f"))

def num_discs_vs_exp(data):
    for d in data:
        d['num_discs'] = int(d['run'].strip('.yaml').split('_')[-1])
    data = compare_same_set(data)
    df = pd.DataFrame(data)
    print_header('Num Discs vs Solved')
    d = df.groupby(['num_discs', 'exp_name']).agg(['mean', 'sum']).pivot_table(columns='exp_name', values=[('solved', 'mean'), ('solved', 'sum')], index='num_discs')
    print(d.to_string(float_format="%.2f"))

    print_header('Num Discs vs FD time')
    df.loc[~df.solved, 'run_time'] = 60
    d = df.groupby(['num_discs', 'exp_name']).agg(['mean', 'median']).pivot_table(columns='exp_name', values=[('total_fd_search_time', 'mean')], index='num_discs')
    print(d.to_string(float_format="%.2f"))
    #d = df.groupby(['num_discs', 'exp_name']).agg(['mean', 'sum']).pivot_table(columns='exp_name', values=[('solved', 'mean'), ('run_time', 'mean')], index='num_discs')
    #print(d.to_string(float_format="%.2f"))

def num_blocks_vs_exp(data):
    for d in data:
        d['num_blocks'], d['num_blockers'], d['max_stack'], _ = map(int, d['run'].strip('.yaml').split('_'))

    data = compare_same_set(data)
    df = pd.DataFrame(data)

    print_header('Num Blocks vs Solved')
    d = df.groupby(['num_blocks', 'exp_name']).agg(['mean', 'sum']).pivot_table(columns='exp_name', values=[('solved', 'mean'), ('solved', 'sum')], index='num_blocks')
    print(d.to_string(float_format="%.2f"))

    print_header('Num Blocks vs Runtime')
    df.loc[~df.solved, 'run_time'] = 60
    d = df.groupby(['num_blocks', 'exp_name']).agg(['mean', 'median']).pivot_table(columns='exp_name', values=[('run_time', 'mean')], index='num_blocks')
    print(d.to_string(float_format="%.2f"))

    print_header('Num Blocks vs Results')
    d = df.groupby(['num_blocks', 'exp_name']).agg(['mean', 'median']).pivot_table(columns='exp_name', values=[('results', 'mean')], index='num_blocks')
    print(d.to_string(float_format="%.2f"))

def remove_probably_infeasible(data):
    assert False, "Depreciated"
    return [d for d in data if not (d.get('sample_time', 0) > 0 and not d['solved'])]

def print_header(st):
    print(('\n' * 2) + ('#' * 10), st, ('#' * 10) + ('\n' * 1))


if __name__ == '__main__':
    import pandas as pd

    data_adaptive = load_results_from_stats('/home/agrobenj/drake-tamp/experiments/hanoi_logs/test/adaptive/', 'adaptive')
    print_header('Adaptive')
    table_compare(data_adaptive)
    num_discs_vs_exp(data_adaptive)
    data_informed = load_results_from_stats('/home/agrobenj/drake-tamp/experiments/hanoi_logs/test/informed/', 'informed')
    print_header('Informed')
    table_compare(data_informed)
    num_discs_vs_exp(data_informed)
    #print_header('Oracle')
    #data_oracle = parse_logs('/home/agrobenj/drake-tamp/experiments/hanoi_logs/train/oracle/', 'oracle')
    #table_compare(data_oracle)
