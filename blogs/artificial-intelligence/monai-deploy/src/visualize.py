from monai.visualize import blend_images, matshow3d
import matplotlib.pyplot as plt
import nibabel as nib
import os

output_image = nib.load(os.path.join('./output/0/0_trans.nii.gz')).get_fdata()
print(output_image.shape)
matshow3d(
    volume=output_image[:,:,200:300],
    fig=None,
    title="output image",
    figsize=(100, 100),
    every_n=5,
    frame_dim=-1,
    show=True,
    # cmap="gray",
)