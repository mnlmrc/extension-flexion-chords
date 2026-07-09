from EFC_learningfMRI import rois, force, betas, searchlight, hrf, geometry, correlation
import EFC_learningfMRI.globals as gl
import time
import argparse

from scripts import activation

def main(args):

    # retrieve the single-trial behavioural metrix in each session and participant
    if args.what == 'behaviour':
        for sn in args.sns:
            for sess in args.session:
                force.calc_behaviour(sn, sess)

    # save the raw and predicted BOLD signal in each participant
    elif args.what == 'save_bold':
        for sn in args.sns:
            hrf.save_bold_rois(sn, args.glm, rois=gl.rois[args.atlas_name])

    # optimise hrf using gridsearch
    elif args.what == 'optimise_hrf':
        for sn in args.sns:
            HRF = hrf.Optimise_HRF(sn, args.glm, rois=gl.rois[args.atlas_name], H='L')
            HRF.gridsearch()
    
    elif args.what == "spm_as_mat7":
        for sn in args.sns:
            betas.save_spm_as_mat7(sn=sn, glm=args.glm)

    # make nifti files for ROI masks in individual space
    elif args.what == "make_cortical_rois":
        for sn in args.sns:
            rois.make_cortical_rois(sn=sn, atlas_name=args.atlas_name, glm=args.glm)
            rois.make_hemispheres(sn=sn, glm=args.glm)


    elif args.what == "make_searchlight":
        for sn in args.sns:
            searchlight.make_searchlight(sn=sn)

    # save activity patterns as 3D cifti files
    elif args.what == "make_cifti_beta":
        for sn in args.sns:
            betas.make_cifti_cortex(sn=sn, glm=args.glm, type='beta')

    elif args.what == "make_cifti_rep_suppr":
        for sn in args.sns:
            betas.make_cifti_cortex(sn=sn, glm=args.glm, type='repetition_suppression')

    # save residuals as 4D cifti file
    elif args.what == "make_cifti_residual":
        for sn in args.sns:
            betas.make_cifti_cortex(sn=sn, glm=args.glm, type='residual')

    # save contrasts vs. resting baseline to 3D cifti
    elif args.what == "make_cifti_contrast":
        for sn in args.sns:
            betas.make_cifti_cortex(sn=sn, glm=args.glm, type='contrast')

    elif args.what == "make_cifti_intercept":
        for sn in args.sns:
            betas.make_cifti_cortex(sn=sn, glm=args.glm, type='intercept')
            
    elif args.what == "make_cifti_psc":
        for sn in args.sns:
            betas.make_cifti_cortex(sn=sn, glm=args.glm, type='psc')

    # save average activity vs. baseline in each ROI
    elif args.what == "roi_contrasts":
        betas.roi_contrasts(sns=args.sns, atlas_name=args.atlas_name, glm=args.glm)

    # make smooth surface cifti file for contrast vs. resting baseline
    elif args.what == "smooth_cifti_contrast":
        for sn in args.sns:
            activation.gifti2cifti_contrasts(sn=sn, glm=args.glm)
        activation.smooth_cifti_contrasts(sns=args.sns, glm=args.glm)
    
    elif args.what == "G_trained_untrained":
        geometry.calc_G(sns=args.sns, glm=args.glm, rois=gl.rois[args.atlas_name], type='trained-untrained')
    
    elif args.what == "G_chord_session":
        geometry.calc_G(sns=args.sns, glm=args.glm, rois=gl.rois[args.atlas_name], type='chord-session', sessions=gl.sessions)
        
    elif args.what == "searchlight_encoding_session":
        geometry.searchlight_encoding(sns=args.sns, glm=args.glm)

    elif args.what == "correlation_between_sessions":
        correlation.correlation_sess(sns=args.sns, glm=args.glm, rois=gl.rois[args.atlas_name], atlas_name=args.atlas_name)
        
    elif args.what == "correlation_between_sessions_rep_suppr":
        pcm.correlation_rep_suppr(sns=args.sns, glm=args.glm, rois=gl.rois[args.atlas_name], atlas_name=args.atlas_name)
    else:
        raise ValueError(f"Unknown command: {args.what}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--sns', nargs='+', type=int, default=[101, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112, 113])
    parser.add_argument('--session', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    parser.add_argument('--rois', nargs='+', type=str, default=gl.rois['ROI'])
    parser.add_argument('--atlas_name', type=str, default='ROI')
    parser.add_argument('--glm', type=int, default=None)

    args = parser.parse_args()

    start = time.time()
    main(args)
    finish = time.time()

    print(f'Execution time:{finish - start} s')