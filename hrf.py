from imaging_pipelines import hrf
import time
import numpy as np
import pandas as pd
import argparse
import os
import globals as gl


def main(args):
    Hem = ['L', 'R']
    struct = ['CortexLeft', 'CortexRight']
    timeseries = ['y_raw', 'y_hat', 'y_adj', 'y_filt']
    if args.what == 'save_timeseries_cifti':
        path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
        masks = [os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}', f'Hem.{H}.nii')
                 for H in Hem]
        print(f'participant {args.sn}, getting timeseries in voxels...')
        cifti_yraw, cifti_yfilt, cifti_yhat, cifti_yadj = hrf.get_timeseries_in_voxels(path_glm, masks, struct)
        nb.save(cifti_yraw, path_glm + '/' + 'y_raw.dtseries.nii')
        nb.save(cifti_yfilt, path_glm + '/' + 'y_filt.dtseries.nii')
        nb.save(cifti_yhat, path_glm + '/' + 'y_hat.dtseries.nii')
        nb.save(cifti_yadj, path_glm + '/' + 'y_adj.dtseries.nii')
    if args.what == 'save_timeseries_parcel':
        path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
        masks = [os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}', f'Hem.{H}.nii')
                 for H in Hem]
        rois = [os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}', f'{args.atlas}.{H}.nii')
                for H in Hem]
        for ts in timeseries:
            print(f'participant {args.sn}, processing {ts} parcels...')
            cifti = nb.load(os.path.join(path_glm, f'{timeseries}.dtseries.nii'))
            cifti_parcel = get_timeseries_in_parcels(path_glm, masks, rois, struct, cifti)
            nb.save(cifti_parcel, os.path.join(gl.baseDir, args.experiment,
                                        f'{gl.glmDir}{args.glm}', f'subj{args.sn}',
                                        f'{args.atlas}.{ts}.ptseries.nii'))
    if args.what == 'save_timeseries_cut':
        path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
        masks = [os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}', f'Hem.{H}.nii')
                 for H in Hem]
        rois = [os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}', f'{args.atlas}.{H}.nii')
                for H in Hem]
        # define onsets (experiment-specific)
        dat = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{args.sn}',
                                       f'{args.experiment}_{args.sn}.dat'), sep='\t')
        pinfo = pd.read_csv(os.path.join(gl.baseDir, args.experiment, 'participants.tsv'), sep='\t')
        runs = pinfo[pinfo['sn'] == args.sn].FuncRuns.reset_index(drop=True)[0].split('.')
        nVols = pinfo[pinfo['sn'] == args.sn].numTR.reset_index(drop=True)[0]
        i = 0
        for BN in dat['BN'].unique():
            if str(BN) in runs:
                if i == 0:
                    at = (dat[dat['BN']==BN].startTRReal).tolist()
                else:
                    at.extend((dat[dat['BN']==BN].startTRReal + int(nVols * i)).tolist())
                i += 1
            else:
                print(f'excluding block {BN}')
        for ts in timeseries:
            print(f'participant {args.sn}, processing {ts} cut parcels...')
            cifti = nb.load(os.path.join(path_glm, f'{ts}.dtseries.nii'))
            cifti_parcel_cut = cut_timeseries_at_onsets(path_glm, masks, rois, struct, cifti, at=at)
            nb.save(cifti_parcel_cut, os.path.join(gl.baseDir, args.experiment,
                                        f'{gl.glmDir}{args.glm}', f'subj{args.sn}',
                                        f'{args.atlas}.{ts}.cut.ptseries.nii'))

    if args.what == 'save_timeseries_cifti_all':
        for sn in args.snS:
            args = argparse.Namespace(
                what='save_timeseries_cifti',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm
            )
            main(args)
    if args.what == 'save_timeseries_parcel_all':
        for sn in args.snS:
            args = argparse.Namespace(
                what='save_timeseries_parcel',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm,
                atlas=args.atlas
            )
            main(args)
    if args.what == 'save_timeseries_cut_all':
        GoNogo = ['go', 'nogo']
        for sn in args.snS:
            for go in GoNogo:
                args = argparse.Namespace(
                    what='save_timeseries_cut',
                    experiment=args.experiment,
                    sn=sn,
                    glm=args.glm,
                    GoNogo=go,
                    atlas=args.atlas
                )
                main(args)
    if args.what == 'save_timeseries_all':
        commands = ['save_timeseries_cifti_all', 'save_timeseries_parcel_all', 'save_timeseries_cut_all']
        for cmd in commands:
            args = argparse.Namespace(
                what=cmd,
                experiment=args.experiment,
                glm=args.glm,
                atlas=args.atlas,
                snS=args.snS
            )
            main(args)

    if args.what == 'save_timeseries_cut_avg':
        data = []
        for sn in args.snS:
            print(f'Processing participant {sn}')
            y_adj = nb.load(os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}', f'subj{sn}',
                                            f'{args.atlas}.y_adj.cut.ptseries.nii'))

            data.append(y_adj_go.dataobj)

            parcel_axis_tmp = y_adj.header.get_axis(1)
            parcel_axis_tmp.affine = None # remove affine to allow concatenation

            if args.snS.index(sn) == 0:
                parcel_axis = parcel_axis_tmp
                row_axis = y_adj.header.get_axis(0)
            else:
                parcel_axis += parcel_axis_tmp

        data = np.hstack(data)

        header = nb.Cifti2Header.from_axes((row_axis, parcel_axis))
        cifti_parcel = nb.Cifti2Image(data, header=header)
        nb.save(cifti_parcel, os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}',
                                           f'{args.atlas}.y_adj.cut.ptseries.nii'))

    if args.what == 'inspect_hrf_params':

        inspect_hrf_params(args.experiment,
                           args.glm,
                           args.sn,
                           args.GoNogo,
                           args.atlas,
                           'S1',
                           [6, 12, 1, 1, 6, 0, 32])


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='efc4')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--snS', nargs='+', default=[102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112], type=int)
    parser.add_argument('--atlas', type=str, default='ROI')
    parser.add_argument('--glm', type=int, default=1)
    parser.add_argument('--roi', type=str, default='S1')
    parser.add_argument('--P', nargs='+', type=int, default=[6, 16, 1, 1, 6, 0, 32])

    args = parser.parse_args()

    main(args)
    end = time.time()
    print(f'Finished in {end - start} seconds')


# # load residuals for prewhitening
# res_img = nb.load(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}', 'ResMS.nii'))
# ResMS = nt.sample_image(res_img, coords[0], coords[1], coords[2], 0)
#
# if stats == 'mean':
#     y_raw = data.mean(axis=1)
# elif stats == 'whiten':
#     y_raw = (data / np.sqrt(ResMS)).mean(axis=1)
# elif stats == 'pca':
#     pass
#
# fdata = SPM.spm_filter(SPM.weight @ data)
# beta = SPM.pinvX @ fdata
# pdata = SPM.design_matrix @ beta
#
#
#
#
#
