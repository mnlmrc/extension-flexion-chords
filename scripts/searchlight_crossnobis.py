from EFC_learningfMRI.searchlight import Searchlight, calc_avg_crossnobis, calc_avg_crossnobis_mnn
import EFC_learningfMRI.globals as gl

if __name__=='__main__':
    sns = [108, 110, 111, 112, 113, 114] # gl.participants
    glm = 3
    searchlight = Searchlight(sns             = sns, 
                              glm             = glm, 
                              multivariate_pw = True,
                              metric_fn       = calc_avg_crossnobis_mnn,
                              out_fname       = 'searchlight_crossnobis')
    searchlight.run()