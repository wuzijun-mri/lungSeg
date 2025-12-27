import numpy as np
import scipy.ndimage as nd
import SimpleITK as sitk
import os

def read_nii(path):
    img_itk = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img_itk)
    return img, img_itk

def save_nii(array, format_itk, path):
    array_itk = sitk.GetImageFromArray(array)
    array_itk.CopyInformation(format_itk)
    sitk.WriteImage(array_itk, path)

def ConnectedCompontFilter(binary_array, thresh):
    labeled_array, num_features = nd.label(binary_array)
    volumes = nd.sum(binary_array, labeled_array, range(num_features + 1))
    # for vol in volumes:
    #     print(vol)
    mask = volumes >= thresh
    mask = mask[labeled_array]
    filtered_mask = mask.astype(np.int8)
    return filtered_mask

eps = np.finfo(np.float64).eps

def chanvese3d(I, init_mask, lr, update_w, mean_w, max_its, alpha):

    I = I.astype('float')
    
    phi = mask2phi(init_mask)
    
    its = 0

    while (its < max_its):
        
        idx = np.where(np.abs(phi) <= update_w)

        inner = np.where((phi<=0) & (phi>=-mean_w))  # interior points c1
        outer = np.where((phi>0) & (phi<=mean_w))  # exterior points c2
        c1 = np.mean(I[inner]) # interior mean
        c2 = np.mean(I[outer]) # exterior mean
        
        #-- calculate dphi/dt
        F = (I[idx]-c1)**2 - (I[idx]-c2)**2  # force from image information
        curvature = get_curvature(phi, idx)  # force from curvature penalty

        dphidt = F/np.max(np.abs(F)) + alpha*curvature  # gradient descent to minimize energy
        dt = lr/(np.max(np.abs(dphidt))+eps)

        #-- evolve the curve
        phi[idx] += dt * dphidt

        its = its + 1

    #-- Get mask from phi(x)
    seg = (phi <= 0) 
    return seg.astype(np.int8), phi, its

def bwdist(a):
    return nd.distance_transform_edt(a == 0)

#-- converts a mask to a SDF
def mask2phi(init_a):
    phi = bwdist(init_a) - bwdist(1-init_a) + init_a - 0.5
    return phi

#-- compute curvature along SDF
def get_curvature(phi,idx):
    z, y, x = idx
    n_points = len(z)

    curvature = np.zeros(n_points, dtype=phi.dtype)

    dimz, dimy, dimx = phi.shape

    zm1 = np.clip(z-1, 0, dimz-1)
    ym1 = np.clip(y-1, 0, dimy-1)
    xm1 = np.clip(x-1, 0, dimx-1)
    zp1 = np.clip(z+1, 0, dimz-1)
    yp1 = np.clip(y+1, 0, dimy-1)
    xp1 = np.clip(x+1, 0, dimx-1)

    phi_z_y_xm1 = phi[z, y, xm1]
    phi_z_y_xp1 = phi[z, y, xp1]
    phi_z_ym1_x = phi[z, ym1, x]
    phi_z_yp1_x = phi[z, yp1, x]
    phi_zm1_y_x = phi[zm1, y, x]
    phi_zp1_y_x = phi[zp1, y, x]

    dx = (phi_z_y_xm1 - phi_z_y_xp1) / 2.0
    dy = (phi_z_ym1_x - phi_z_yp1_x) / 2.0
    dz = (phi_zm1_y_x - phi_zp1_y_x) / 2.0

    dxx = phi_z_y_xm1 - 2*phi[z,y,x] + phi_z_y_xp1
    dyy = phi_z_ym1_x - 2*phi[z,y,x] + phi_z_yp1_x
    dzz = phi_zm1_y_x - 2*phi[z,y,x] + phi_zp1_y_x

    dx2 = dx**2
    dy2 = dy**2
    dz2 = dz**2

    phi_z_ym1_xm1 = phi[z, ym1, xm1]
    phi_z_yp1_xp1 = phi[z, yp1, xp1]
    phi_z_ym1_xp1 = phi[z, ym1, xp1]
    phi_z_yp1_xm1 = phi[z, yp1, xm1]
    phi_zp1_y_xm1 = phi[zp1, y, xm1]
    phi_zm1_y_xp1 = phi[zm1, y, xp1]
    phi_zp1_y_xp1 = phi[zp1, y, xp1]
    phi_zm1_y_xm1 = phi[zm1, y, xm1]
    phi_zp1_ym1_x = phi[zp1, ym1, x]
    phi_zm1_yp1_x = phi[zm1, yp1, x]
    phi_zp1_yp1_x = phi[zp1, yp1, x]
    phi_zm1_ym1_x = phi[zm1, ym1, x]

    dxy = (phi_z_ym1_xm1 + phi_z_yp1_xp1 - phi_z_ym1_xp1 - phi_z_yp1_xm1) / 4.0
    dxz = (phi_zp1_y_xm1 + phi_zm1_y_xp1 - phi_zp1_y_xp1 - phi_zm1_y_xm1) / 4.0
    dyz = (phi_zp1_ym1_x + phi_zm1_yp1_x - phi_zp1_yp1_x - phi_zm1_ym1_x) / 4.0

    grad_mag_sq = dx2 + dy2 + dz2 + eps 
    curvature = (dxx*(dy2 + dz2) + dyy*(dx2 + dz2) + dzz*(dx2 + dy2)
                - 2*dx*dy*dxy - 2*dx*dz*dxz - 2*dy*dz*dyz) / (grad_mag_sq)

    return curvature


# MR images
mri_dir = "PATH_TO_TRAINING_MRI_DATA"
# prediction
infer_dir = "PATH_TO_PREDICTION"
# refined pseudo label
refined_dir = "PATH_TO_SAVE_PSEUDO_LABEL"
if not os.path.exists(refined_dir):
    os.makedirs(refined_dir)

for filename in sorted(os.listdir(infer_dir)):
    refined_path = os.path.join(refined_dir, filename)
    print(filename)
    #-- read prediction and MR image
    pred_path = os.path.join(infer_dir, filename)
    mri_path = os.path.join(mri_dir, filename)
    pred, pred_itk = read_nii(pred_path)
    mri, _ = read_nii(mri_path)
    #-- connected component filer
    pred_cc = ConnectedCompontFilter(pred, 150000)
    #-- Chan-Vese based refinement
    lr = 0.25; max_iter = 100; alpha = 0.1; delta_w=5; mean_w=5
    pred_refined, phi, its = chanvese3d(mri, pred_cc, lr, delta_w, mean_w, max_iter, alpha)
    pred_refined = ConnectedCompontFilter(pred_refined, 150000)
    save_nii(pred_refined, pred_itk, refined_path)
