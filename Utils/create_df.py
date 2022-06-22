import numpy as np
import pandas as pd

def create_df_plots(sketch_sizes, n_rep, sketch_type, mse_test, times):

    n_sizes = len(sketch_sizes)

    sketch_types = []

    for i in range(n_sizes * n_rep):
        sketch_types.append(sketch_type)
 
    sketch_typess = np.asarray(sketch_types)
 
    sketch_sizes_concat = np.empty((0))
    for i_s, s in enumerate(sketch_sizes):
        sketch_sizes_concat = np.concatenate((sketch_sizes_concat, s * np.ones(n_rep)))

    
    sketches_s = np.empty((0))
    sketches = np.empty((0))
    test_mse = np.empty((0))
    timess = np.empty((0))

    sketches_s = np.concatenate((sketches_s, sketch_sizes_concat))
    sketches = np.concatenate((sketches, sketch_typess))
    test_mse = np.concatenate((test_mse, mse_test.flatten()))
    timess = np.concatenate((timess, times.flatten()))

    dic = {'Feature map size': sketches_s, 'Sketch type': sketches, 'Test MSE': test_mse, 'Training times in s': timess}
    df = pd.DataFrame(data=dic)
    
    return df