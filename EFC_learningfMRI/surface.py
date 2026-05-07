import os
import globals.path as pth
import globals.imaging as im
import nitools as nt
import nibabel as nb
import numpy as np
import argparse
from util.util import get_trained_and_untrained
import globals.design as dn
#from EFC_learningfMRI.util import get_trained_and_untrained

def gifti2cifti_contrasts(sn, glm):
    print(f'Processing participant {sn}')
    path = os.path.join(pth.baseDir, pth.surfDir, f'subj{sn}')
    giftis = [path + '/' + f'glm{glm}.con.{H}.func.gii' for H in im.Hem]
    cifti_img = nt.join_giftis_to_cifti(giftis)
    nb.save(cifti_img, path + '/' + f'glm{glm}.con.dscalar.nii')


def smooth_cifti_contrasts(sns, glm):
    data = []
    for sn in sns:
        print(f'Processing participant {sn}')
        img = os.path.join(pth.baseDir, pth.surfDir, f'subj{sn}', f'glm{glm}.con.dscalar.nii')
        cifti_img = nb.load(img)

        chords = get_trained_and_untrained(sn)
        trained, untrained = chords[:4], chords[4:]

        row_axis = cifti_img.header.get_axis(0).name
        sess = []

        if glm == 2:
            for rep in [1, 2]:
                for s in dn.sessions:
                    sess.append([col for col in row_axis
                                 if f'sess{s:02d}' in col
                                 and col.split('_')[1].split(',')[0] in trained
                                 and int(col.split(',')[2].split('.')[0]) == rep])
                    sess.append([col for col in row_axis
                                 if f'sess{s:02d}' in col
                                 and col.split('_')[1].split(',')[0] in untrained
                                 and int(col.split(',')[2].split('.')[0]) == rep])

        elif glm == 1:
            for s in dn.sessions:
                sess.append([col for col in row_axis
                             if f'sess{s:02d}' in col
                             and col.split('_')[1].split(',')[0] in trained])
                sess.append([col for col in row_axis
                             if f'sess{s:02d}' in col
                             and col.split('_')[1].split(',')[0] in untrained])
        else:
            raise ValueError(f"GLM {glm} not found")

        data_tmp = cifti_img.get_fdata()
        dataS = []
        for s in sess:
            im = np.array([x in s for x in row_axis])
            dataS.append(np.array(data_tmp[im]).mean(axis=0))

        data.append(np.vstack(dataS))

        if sns.index(sn) == 0:
            brain_axis = cifti_img.header.get_axis(1)

    data = np.array(data).mean(axis=0)

    if glm == 2:
        row_axis_new = nb.cifti2.ScalarAxis(
            ['sess03,trained,1',
             'sess03,untrained,1',
             'sess09,trained,1',
             'sess09,untrained,1',
             'sess23,trained,1',
             'sess23,untrained,1',
             'sess03,trained,2',
             'sess03,untrained,2',
             'sess09,trained,2',
             'sess09,untrained,2',
             'sess23,trained,2',
             'sess23,untrained,2'])
    elif glm == 1:
        row_axis_new = nb.cifti2.ScalarAxis(
            ['sess03,trained',
             'sess03,untrained',
             'sess09,trained',
             'sess09,untrained',
             'sess23,trained',
             'sess23,untrained'])
    header = nb.Cifti2Header.from_axes((row_axis_new, brain_axis))
    cifti_img = nb.Cifti2Image(dataobj=data, header=header)
    nb.save(cifti_img, os.path.join(pth.baseDir, pth.surfDir, f'glm{glm}.con.session.dscalar.nii'))
    nt.smooth_cifti(os.path.join(pth.baseDir, pth.surfDir, f'glm{glm}.con.session.dscalar.nii'),
                    os.path.join(pth.baseDir, pth.surfDir, f'glm{glm}.con.session.smooth.dscalar.nii'),
                    os.path.join(pth.atlasDir, 'fs_LR.32k.L.flat.surf.gii'),
                    os.path.join(pth.atlasDir, 'fs_LR.32k.R.flat.surf.gii'))



# def main(args):
#     Hem = ['L', 'R']
#     if args.what == 'gifti2cifti_betas':
#         print(f'Processing participant {args.sn}')
#         path = os.path.join(pth.baseDir, pth.surfDir, f'subj{args.sn}')
#         giftis = [path + '/' + f'glm{args.glm}.{args.dtype}.{H}.func.gii' for H in Hem]
#         cifti_img = nt.join_giftis_to_cifti(giftis)
#         nb.save(cifti_img, path + '/' + f'glm{args.glm}.{args.dtype}.dscalar.nii')
#     if args.what=='smooth_cifti_betas':
#         data, dataDiff = [], []
#         for sn in args.sns:
#             print(f'Processing participant {sn}')
#             img = os.path.join(pth.baseDir, pth.surfDir, f'subj{sn}', f'glm{args.glm}.{args.dtype}.dscalar.nii')
#             cifti_img = nb.load(img)
#
#             chords = get_trained_and_untrained(sn)
#             trained, untrained = chords[:4], chords[4:]
#
#             row_axis = cifti_img.header.get_axis(0).name
#             sess = []
#
#             if args.glm == 2:
#                 for rep in [1, 2]:
#                     for s in gl.sessions:
#                         sess.append([col for col in row_axis
#                                      if f'sess{s:02d}' in col
#                                      and col.split('_')[1].split(',')[0] in trained
#                                      and int(col.split(',')[2].split('.')[0])==rep])
#                         sess.append([col for col in row_axis
#                                      if f'sess{s:02d}' in col
#                                      and col.split('_')[1].split(',')[0] in untrained
#                                      and int(col.split(',')[2].split('.')[0])==rep])
#             if args.glm == 3:
#                 for s in gl.sessions:
#                     sess.append([col for col in row_axis
#                                  if f'sess{s:02d}' in col
#                                  and col.split('_')[1].split(',')[0] in trained])
#                     sess.append([col for col in row_axis
#                                  if f'sess{s:02d}' in col
#                                  and col.split('_')[1].split(',')[0] in untrained])
#
#             data_tmp = cifti_img.get_fdata()
#             dataS = []
#             for s in sess:
#                 im = np.array([x in s for x in row_axis])
#                 dataS.append(np.array(data_tmp[im]).mean(axis=0))
#
#             #dataDiff.append(np.vstack([dataS[0] - dataS[1], dataS[2] - dataS[3], dataS[4] - dataS[5]]))
#             data.append(np.vstack(dataS))
#
#             if args.sns.index(sn) == 0:
#                 brain_axis = cifti_img.header.get_axis(1)
#
#         data = np.array(data).mean(axis=0)
#         #dataDiff = np.array(dataDiff).mean(axis=0)
#
#         if args.glm == 2:
#             row_axis_new = nb.cifti2.ScalarAxis(
#                 ['sess03,trained,1',
#                  'sess03,untrained,1',
#                  'sess09,trained,1',
#                  'sess09,untrained,1',
#                  'sess23,trained,1',
#                  'sess23,untrained,1',
#                  'sess03,trained,2',
#                  'sess03,untrained,2',
#                  'sess09,trained,2',
#                  'sess09,untrained,2',
#                  'sess23,trained,2',
#                  'sess23,untrained,2'])
#         if args.glm == 3:
#             row_axis_new = nb.cifti2.ScalarAxis(
#                 ['sess03,trained',
#                  'sess03,untrained',
#                  'sess09,trained',
#                  'sess09,untrained',
#                  'sess23,trained',
#                  'sess23,untrained'])
#         header = nb.Cifti2Header.from_axes((row_axis_new, brain_axis))
#         cifti_img = nb.Cifti2Image(
#             dataobj=data,  # Stack them along the rows (adjust as needed)
#             header=header,  # Use one of the headers (may need to modify)
#         )
#         nb.save(cifti_img, os.path.join(pth.baseDir, gl.surfDir, f'glm{args.glm}.{args.dtype}.session.dscalar.nii'))
#         nt.smooth_cifti(os.path.join(gl.baseDir, gl.surfDir, f'glm{args.glm}.{args.dtype}.session.dscalar.nii'),
#                         os.path.join(gl.baseDir, gl.surfDir, f'glm{args.glm}.{args.dtype}.session.smooth.dscalar.nii'),
#                         os.path.join(gl.atlasDir, 'fs_LR.32k.L.flat.surf.gii'),
#                         os.path.join(gl.atlasDir, 'fs_LR.32k.R.flat.surf.gii'))
#
#         # row_axis = nb.cifti2.ScalarAxis(['sess03', 'sess09', 'sess23'])
#         # header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
#         # cifti_img = nb.Cifti2Image(
#         #     dataobj=dataDiff,  # Stack them along the rows (adjust as needed)
#         #     header=header,  # Use one of the headers (may need to modify)
#         # )
#         # nb.save(cifti_img, os.path.join(gl.baseDir, gl.surfDir, f'glm{args.glm}.{args.dtype}.trained_vs_untrained.dscalar.nii'))
#         # nt.smooth_cifti(os.path.join(gl.baseDir, gl.surfDir, f'glm{args.glm}.{args.dtype}.trained_vs_untrained.dscalar.nii'),
#         #                 os.path.join(gl.baseDir, gl.surfDir, f'glm{args.glm}.{args.dtype}.trained_vs_untrained.smooth.dscalar.nii'),
#         #                 os.path.join(gl.atlasDir, 'fs_LR.32k.L.flat.surf.gii'),
#         #                 os.path.join(gl.atlasDir, 'fs_LR.32k.R.flat.surf.gii'))
#
#
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('what', nargs='?', default=None)
#     parser.add_argument('--sn', type=int, default=None)
#     parser.add_argument('--sns', nargs='+', default=[101, 102, 103, 104, 105, 106, 107,])
#     parser.add_argument('--dtype', type=str, default='con')
#     parser.add_argument('--glm', type=int, default=1)
#
#     args = parser.parse_args()
#
#     main(args)