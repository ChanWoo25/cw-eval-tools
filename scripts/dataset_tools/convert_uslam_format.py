import os
import argparse
from pathlib import Path
import numpy as np

from colorama import init, Fore, Style
init(autoreset=True)

def extract(source_fn, out_filename):
    fout = open(out_filename, 'w')
    fout.write('# timestamp tx ty tz qx qy qz qw\n')

    with open(source_fn, 'rb') as fin:
        data = np.genfromtxt(fin, delimiter=",", comments='#')
        for l in data:
            fout.write('%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\n' %
                       (l[0]/1e9, l[1], l[2], l[3], l[4], l[5], l[6], l[7]))
    fout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Extracts pose from a csv file in ASL format to space separated format.
    Quaternion is ordered as [x y z w]
    ''')
    parser.add_argument('--single-file', type=str, default=None)
    parser.add_argument('--input_dir', default='')
    args = parser.parse_args()

    print(Fore.GREEN+'RUN::convert uslam_traj format to eval format')
    if args.single_file is not None:
        source_fn = Path(args.single_file)
        source_dir = source_fn.parent
        result_fn = source_dir / 'estimate.txt'
        extract(source_fn, result_fn)
        print(' - source_fn: %s' % source_fn)
        print(' - result_fn: %s' % result_fn)
        exit(0)

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
