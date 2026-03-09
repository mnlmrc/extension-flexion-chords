from imaging import rois, betas, searchlight, surface, hrf, pcm, geometry
from behavioural import force
import globals.imaging as im
import globals.design as dn
import time
import argparse

def main(args):
    if args.what == 'behaviour':
        for sn in args.sns:
            for sess in args.session:
                force.calc_behaviour(sn, sess)
    elif args.what == "save_BOLD":
        for sn in args.sns:
            hrf.save_BOLD(sn=sn, glm=args.glm,)
    elif args.what == "optimise_hrf":
        for sn in args.sns:
            hrf.optimise_hrf(sn=sn, glm=args.glm)
    elif args.what == "calc_R2_adj_hat":
        hrf.calc_R2_adj_hat(sns=args.sns, glm=args.glm, atlas_name=args.atlas_name)
    elif args.what == "spm_as_mat7":
        for sn in args.sns:
            betas.save_spm_as_mat7(sn=sn, glm=args.glm)
    elif args.what == "make_cortical_rois":
        for sn in args.sns:
            rois.make_cortical_rois(sn=sn, atlas_name=args.atlas_name, glm=args.glm)
            rois.make_hemispheres(sn=sn, glm=args.glm)
    elif args.what == "make_searchlight":
        for sn in args.sns:
            searchlight.make_searchlight(sn=sn)
    elif args.what == "make_cifti_beta":
        for sn in args.sns:
            betas.make_cifti(sn=sn, glm=args.glm, type='beta')
    elif args.what == "make_cifti_repetition_suppression":
        for sn in args.sns:
            betas.make_cifti(sn=sn, glm=args.glm, type='repetition_suppression')
    elif args.what == "make_cifti_residual":
        for sn in args.sns:
            betas.make_cifti(sn=sn, glm=args.glm, type='residual')
    elif args.what == "make_cifti_contrast":
        for sn in args.sns:
            betas.make_cifti(sn=sn, glm=args.glm, type='contrast')
    elif args.what == "make_cifti_intercept":
        for sn in args.sns:
            betas.make_cifti(sn=sn, glm=args.glm, type='intercept')
    elif args.what == "make_cifti_psc":
        for sn in args.sns:
            betas.make_cifti(sn=sn, glm=args.glm, type='psc')
    elif args.what == "roi_contrasts":
        betas.roi_contrasts(sns=args.sns, atlas_name=args.atlas_name, glm=args.glm)
    elif args.what == "gifti2cifti_contrast":
        for sn in args.sns:
            surface.gifti2cifti_contrasts(sn=sn, glm=args.glm)
    elif args.what == "smooth_cifti_contrast":
        surface.smooth_cifti_contrasts(sns=args.sns, glm=args.glm)
    elif args.what == "G_trained_untrained":
        geometry.calc_G(sns=args.sns, glm=args.glm, rois=im.rois[args.atlas_name], type='trained-untrained')
    elif args.what == "G_chord_session":
        geometry.calc_G(sns=args.sns, glm=args.glm, rois=im.rois[args.atlas_name], type='chord-session',
                        sessions=dn.sessions)
    elif args.what == "searchlight_session":
        pcm.pcm_searchlight_sess(sns=args.sns, glm=args.glm)
    elif args.what == "correlation_between_sessions":
        pcm.correlation(sns=args.sns, glm=args.glm, rois=im.rois[args.atlas_name], atlas_name=args.atlas_name)
    elif args.what == "correlation_between_sessions_rep_suppr":
        pcm.correlation_rep_suppr(sns=args.sns, glm=args.glm, rois=im.rois[args.atlas_name], atlas_name=args.atlas_name)
    else:
        raise ValueError(f"Unknown command: {args.what}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--sns', nargs='+', type=int, default=[101, 102, 103, 104, 105, 106, 107, 108])
    parser.add_argument('--session', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    parser.add_argument('--atlas_name', type=str, default='ROI')
    parser.add_argument('--glm', type=int, default=1)

    args = parser.parse_args()

    start = time.time()
    main(args)
    finish = time.time()

    print(f'Execution time:{finish - start} s')