import os
import numpy as np
import scipy.ndimage as nd
import SimpleITK as sitk

def ConnectedCompontFilter(binary_array, thresh):
    labeled_array, num_features = nd.label(binary_array)
    volumes = nd.sum(binary_array, labeled_array, range(num_features + 1))
    # for vol in volumes:
    #     print(vol)
    mask = volumes >= thresh
    mask = mask[labeled_array]
    filtered_mask = mask.astype(np.int8)
    return filtered_mask

def save_nii(array, format_itk, path):
    array_itk = sitk.GetImageFromArray(array)
    array_itk.CopyInformation(format_itk)
    sitk.WriteImage(array_itk, path)

def read_nii(path):
    img_itk = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img_itk)
    return img, img_itk

MC_dir = "PATH_TO_DROPOUT_RESULT"
infer_dir = "PATH_TO_PREDICTION_RESULT"

uncertainty_dict = {}
for fileName in sorted(os.listdir(MC_dir)):
    if '_dp1.nii.gz' not in fileName:
        continue
    # dropout probaility map (K = 5)
    dp1_path = os.path.join(MC_dir, fileName)
    dp2_path = os.path.join(MC_dir, fileName.replace('dp1.nii.gz', 'dp2.nii.gz'))
    dp3_path = os.path.join(MC_dir, fileName.replace('dp1.nii.gz', 'dp3.nii.gz'))
    dp4_path = os.path.join(MC_dir, fileName.replace('dp1.nii.gz', 'dp4.nii.gz'))
    dp5_path = os.path.join(MC_dir, fileName.replace('dp1.nii.gz', 'dp5.nii.gz'))
    # prediction
    pred_path = os.path.join(infer_dir, fileName.replace('_dp1.nii.gz', '.nii.gz'))
    #-- read data
    dp1, dp1_itk = read_nii(dp1_path)
    dp2, _ = read_nii(dp2_path)
    dp3, _ = read_nii(dp3_path)
    dp4, _ = read_nii(dp4_path)
    dp5, _ = read_nii(dp5_path)
    pred, _ = read_nii(pred_path)
    #-- get filtered prediction
    pred = ConnectedCompontFilter(pred, 150000)
    #-- calculate uncertainty within the filtered prediction
    prob_stacked = np.stack((dp1, dp2, dp3, dp4, dp5), axis=0)
    variance = np.var(prob_stacked, axis=0, ddof=1)
    uncertainty = variance[pred > 0]
    uncertainty_dict[fileName] = np.mean(uncertainty)
    # print uncertainty
    print(fileName, np.mean(uncertainty))
