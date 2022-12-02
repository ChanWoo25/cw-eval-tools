#!/usr/bin/env python2

import os
import argparse

import numpy as np


def extract(gt, out_filename):
    fout = open(out_filename, 'w')
    fout.write('# timestamp tx ty tz qx qy qz qw\n')
    with open(gt, 'rb') as fin:
        data = np.genfromtxt(fin, delimiter=",", skip_header=1)
        for l in data:
            fout.write('%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\n' %
                       (l[0]/1e9, l[1], l[2], l[3], l[4], l[5], l[6], l[7]))
    fout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Extracts pose from a csv file in ASL format to space separated format.
    Quaternion is ordered as [x y z w]
    ''')
    parser.add_argument('--input_dir', default='', required=True)
    args = parser.parse_args()

    from os import listdir
    dirs = [f for f in listdir(args.input_dir)]

    cnt = 0
    if len(dirs) == 1:
        intput_fn = os.path.join(args.input_dir, dirs[0], 'traj_es.csv')
        output_fn = 'stamped_traj_estimate.txt'
        output_fn = os.path.join(args.input_dir, output_fn)
        extract(intput_fn, output_fn)
    elif len(dirs) > 1:
        for i in range(len(dirs)):
            dirpath = os.path.join(args.input_dir, dirs[i])
            if not os.path.isdir(dirpath):
                continue

            intput_fn = os.path.join(args.input_dir, dirs[i], 'traj_es.csv')
            output_fn = 'stamped_traj_estimate%d.txt' % cnt
            output_fn = os.path.join(args.input_dir, output_fn)
            cnt += 1

            print('Convert From ' + intput_fn)
            extract(intput_fn, output_fn)
            print('Convert  To  ' + output_fn)
