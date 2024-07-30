# import os
# import warnings
#
# import numpy as np
#
#
# def load_mov_block(filename):
#     """
#     load .mov file of one block
#
#     :return:
#     """
#
#     try:
#         with open(filename, 'rt') as fid:
#             trial = 0
#             A = []
#             for line in fid:
#                 if line.startswith('Trial'):
#                     trial_number = int(line.split(' ')[1])
#                     trial += 1
#                     if trial_number != trial:
#                         warnings.warn('Trials out of sequence')
#                         trial = trial_number
#                     A.append([])
#                 else:
#                     # Convert line to a numpy array of floats and append to the last trial's list
#                     data = np.fromstring(line, sep=' ')
#                     if A:
#                         A[-1].append(data)
#                     else:
#                         # This handles the case where a data line appears before any 'Trial' line
#                         warnings.warn('Data without trial heading detected')
#                         A.append([data])
#
#             # Convert all sublists to numpy arrays
#             mov = [np.array(trial_data) for trial_data in A]
#             # # vizForce = [np.array(trial_data)[:, 9:] for trial_data in A]
#             # state = [np.array(trial_data) for trial_data in A]
#
#     except IOError as e:
#         raise IOError(f"Could not open {filename}") from e
#
#     return mov
#
#
# def get_mov(path, participant_id, nblocks):
#
#     sn = int(''.join([c for c in participant_id if c.isdigit()]))
#
#     mov = list()
#
#     for bl in range(nblocks):
#         block = '%02d' % int(bl + 1)
#
#         filename = os.path.join(path, f'{experiment}_{sn}_{block}.mov')
#
#         mov.append(load_mov_block(filename))
#
#     return mov