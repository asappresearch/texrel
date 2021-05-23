"""
given csv from runners/run_measuring_comp_section7.py,
add gen_err column
also adds ptre
"""
import csv
import argparse


def run(args):
    with open(args.in_csv, 'r') as f:
        dict_reader = csv.DictReader(f)
        fieldnames = dict_reader.fieldnames
        rows = list(dict_reader)
    for row in rows:
        row['gen_err'] = '%.3f' % (float(row['same_acc']) - float(row['new_acc']))
        row['ptre'] = '%.3f' % (float(row['tre']) / float(row['prec']))
    with open(args.out_csv, 'w') as f:
        dict_writer = csv.DictWriter(f, fieldnames=fieldnames + ['gen_err', 'ptre'])
        dict_writer.writeheader()
        for row in rows:
            dict_writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-csv', type=str, required=True)
    parser.add_argument('--out-csv', type=str, required=True)
    args = parser.parse_args()
    run(args)
