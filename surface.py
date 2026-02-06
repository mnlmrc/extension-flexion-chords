import os
import globals as gl
import nitools as nt
import nibabel as nb
import numpy as np
import argparse
from util import get_trained_and_untrained

def main(args):
    Hem = ['L', 'R']
    if args.what == 'gifti2cifti':
        print(f'Processing participant {args.sn}')
        path = os.path.join(gl.baseDir, gl.surfDir, f'subj{args.sn}')
        giftis = [path + '/' + f'glm{args.glm}.{args.dtype}.{H}.func.gii' for H in Hem]
        cifti_img = nt.join_giftis_to_cifti(giftis)
        nb.save(cifti_img, path + '/' + f'glm{args.glm}.{args.dtype}.dscalar.nii')
    if args.what=='smooth_cifti':
        data, dataDiff = [], []
        for sn in args.snS:
            print(f'Processing participant {sn}')
            img = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}', f'glm{args.glm}.{args.dtype}.dscalar.nii')
            cifti_img = nb.load(img)

            chords = get_trained_and_untrained(sn)
            trained, untrained = chords[:4], chords[4:]

            row_axis = cifti_img.header.get_axis(0).name
            sess = []
            sess.append([col for col in row_axis if 'con_3' in col and col.split(',')[1].split('.')[0] in trained])
            sess.append([col for col in row_axis if 'con_3' in col and col.split(',')[1].split('.')[0] in untrained])
            sess.append([col for col in row_axis if 'con_9' in col and col.split(',')[1].split('.')[0] in trained])
            sess.append([col for col in row_axis if 'con_9' in col and col.split(',')[1].split('.')[0] in untrained])
            sess.append([col for col in row_axis if 'con_23' in col and col.split(',')[1].split('.')[0] in trained])
            sess.append([col for col in row_axis if 'con_23' in col and col.split(',')[1].split('.')[0] in untrained])

            data_tmp = cifti_img.get_fdata()
            dataS = []
            for s in sess:
                im = np.array([x in s for x in row_axis])
                dataS.append(np.array(data_tmp[im]).mean(axis=0))

            dataDiff.append(np.vstack([dataS[0] - dataS[1], dataS[2] - dataS[3], dataS[4] - dataS[5]]))
            data.append(np.vstack(dataS))

            if args.snS.index(sn) == 0:
                brain_axis = cifti_img.header.get_axis(1)

        data = np.array(data).mean(axis=0)
        dataDiff = np.array(dataDiff).mean(axis=0)

        row_axis = nb.cifti2.ScalarAxis(['sess1,trained','sess1,untrained', 'sess2,trained','sess2,untrained',
                                         'sess3,trained','sess3,untrained'])
        header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
        cifti_img = nb.Cifti2Image(
            dataobj=data,  # Stack them along the rows (adjust as needed)
            header=header,  # Use one of the headers (may need to modify)
        )
        nb.save(cifti_img, os.path.join(gl.baseDir, gl.surfDir, f'glm{args.glm}.{args.dtype}.session.dscalar.nii'))
        nt.smooth_cifti(os.path.join(gl.baseDir, gl.surfDir, f'glm{args.glm}.{args.dtype}.session.dscalar.nii'),
                        os.path.join(gl.baseDir, gl.surfDir, f'glm{args.glm}.{args.dtype}.session.smooth.dscalar.nii'),
                        os.path.join(gl.atlasDir, 'fs_LR.32k.L.flat.surf.gii'),
                        os.path.join(gl.atlasDir, 'fs_LR.32k.R.flat.surf.gii'))

        row_axis = nb.cifti2.ScalarAxis(['sess1', 'sess2', 'sess3'])
        header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
        cifti_img = nb.Cifti2Image(
            dataobj=dataDiff,  # Stack them along the rows (adjust as needed)
            header=header,  # Use one of the headers (may need to modify)
        )
        nb.save(cifti_img, os.path.join(gl.baseDir, gl.surfDir, f'glm{args.glm}.{args.dtype}.trained_vs_untrained.dscalar.nii'))
        nt.smooth_cifti(os.path.join(gl.baseDir, gl.surfDir, f'glm{args.glm}.{args.dtype}.trained_vs_untrained.dscalar.nii'),
                        os.path.join(gl.baseDir, gl.surfDir, f'glm{args.glm}.{args.dtype}.trained_vs_untrained.smooth.dscalar.nii'),
                        os.path.join(gl.atlasDir, 'fs_LR.32k.L.flat.surf.gii'),
                        os.path.join(gl.atlasDir, 'fs_LR.32k.R.flat.surf.gii'))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--snS', nargs='+', default=[101, 102, 103, 104, 105, 106, 107,])
    parser.add_argument('--dtype', type=str, default='con')
    parser.add_argument('--glm', type=int, default=1)

    args = parser.parse_args()

    main(args)