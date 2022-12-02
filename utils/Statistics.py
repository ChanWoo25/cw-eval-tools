import yaml
import numpy as np
import torch

import os
import sys
print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import SO3

class Statistics:
    device_ = torch.device('cuda:%d'%(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu')
    eps_ = 1e-8

    @classmethod
    def process_1d(cls, data:np.ndarray) -> dict:
        """Compute translation error statistics and return dictionary about it"""
        assert(data.size > 0)
        data.squeeze()
        assert(data.ndim == 1)
        data:torch.Tensor = torch.from_numpy(data).to(device=cls.device_)

        rmse = torch.sqrt(torch.dot(data, data) / data.size(0))
        std, mean = torch.std_mean(data)
        median = data.median()

        return {
            'rmse': rmse.cpu().item(),
            'median': median.cpu().item(),
            'std': std.cpu().item(),
            'mean': mean.cpu().item(),
            'min': data.min().cpu().item(),
            'max': data.max().cpu().item(),
            'size': data.size(0)
        }

# def update_and_save_stats(new_stats, label, yaml_filename):
#     stats = dict()
#     if os.path.exists(yaml_filename):
#         stats = yaml.load(open(yaml_filename, 'r'), Loader=yaml.FullLoader)
#     stats[label] = new_stats

#     with open(yaml_filename, 'w') as outfile:
#         outfile.write(yaml.dump(stats, default_flow_style=False))

#     return


# def compute_and_save_statistics(data_vec, label, yaml_filename):
#     new_stats = compute_statistics(data_vec)
#     update_and_save_stats(new_stats, label, yaml_filename)

#     return new_stats


# def write_tex_table(list_values, rows, cols, outfn):
#     '''
#     write list_values[row_idx][col_idx] to a table that is ready to be pasted
#     into latex source

#     list_values is a list of row values

#     The value should be string of desired format
#     '''

#     assert len(rows) >= 1
#     assert len(cols) >= 1

#     with open(outfn, 'w') as f:
#         # write header
#         f.write('      &      ')
#         for col_i in cols[:-1]:
#             f.write(col_i + ' & ')
#         f.write(' ' + cols[-1]+'\n')

#         # write each row
#         for row_idx, row_i in enumerate(list_values):
#             f.write(rows[row_idx] + ' &     ')
#             row_values = list_values[row_idx]
#             for col_idx in range(len(row_values) - 1):
#                 f.write(row_values[col_idx] + ' & ')
#             f.write(' ' + row_values[-1]+' \n')
