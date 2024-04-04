import os
from monai.deploy.core import Operator, Image, IOType, DataPath
import monai.deploy.core as md
import nibabel as nib
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    Resized,
    NormalizeIntensityd,
    ScaleIntensityd,
    Activationsd,
    AsDiscreted,
    Invertd,
    SaveImaged
)
import logging
from monai.deploy.operators.monai_seg_inference_operator import InMemImageReader, MonaiSegInferenceOperator
from monai.deploy.operators import NiftiDataLoader
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import SegResNet
import torch
import numpy as np


@md.output('image', np.ndarray, IOType.IN_MEMORY)
@md.output('model', str, IOType.IN_MEMORY)
class PreprocessNiftiOperator(Operator):
    def __init__(self):
        super().__init__()
        
    def compute(self, op_input, op_output, context):
        input_path = op_input.get().path 
        logging.info(input_path)
        files = os.listdir(input_path)

        if 'model.pt' not in files:
            logging.error('Cannot find model.pt file in the input path')
            return
        
        input_file = files[0] # ct.nii.gz file
        nifti_input = nib.load(os.path.join(input_path,input_file)).get_fdata()
        op_output.set(nifti_input, 'image')
        op_output.set(os.path.join(input_path, 'model.pt'), 'model')

@md.input('image', np.ndarray, IOType.IN_MEMORY)
@md.input('model', str, IOType.IN_MEMORY)
@md.output('output_msg', str, IOType.IN_MEMORY)
class SegInferenceOperator(Operator):
    def __init__(self, roi_size=96, 
                       pre_transforms = True, 
                       post_transforms = True,
                       model_name = 'SegResNet'):
        super().__init__()
    
    def compute(self, op_input, op_output, context):
        input_image = op_input.get("image") #ndarray
        input_image = np.expand_dims(input_image, axis=-1)
        # input_image = torch.tensor(input_image).float()
        
        # We're not using standard MonaiSegInferenceOperator because of its deep integration 
        # with DICOM style metadata rendering inflexible for NIFTI data. Moreover, it loads 
        # monai.core.model.Model skeleton from app context rather than torch.nn.Module
        
        net = SegResNet(spatial_dims= 3,
                        in_channels= 1,
                        out_channels= 105,
                        init_filters= 32,
                        blocks_down= [1,2,2,4],
                        blocks_up= [1,1,1],
                        dropout_prob= 0.2)
        
        
        #Load the model with pretrained checkpoint extracted from InputContext
        logging.info(os.getcwd())
        net.load_state_dict(torch.load(op_input.get('model')))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)
        
        # Define the list of transforms to be applied
        pixdim = [3.0, 3.0, 3.0]
        self.pre_transforms = Compose([
            EnsureChannelFirstd(keys='image', channel_dim=-1),
            Orientationd(keys='image', axcodes='RAS'),
            # Resized(keys="image", spatial_size=(208,208,208), mode="trilinear", align_corners=True),
            Spacingd(keys='image', pixdim=pixdim, mode='bilinear'),
            NormalizeIntensityd(keys='image', nonzero=True),
            ScaleIntensityd(keys='image', minv=-1.0, maxv=1.0)
        ])
        self.post_transforms = Compose([
            Activationsd(keys='pred', sigmoid=True),
            AsDiscreted(keys='pred', argmax=True),
            Invertd(keys='pred', transform=self.pre_transforms, orig_keys='image'),
            SaveImaged(keys='pred', output_dir='/home/aac/monai-2/output', meta_keys='pred_meta_dict')
        ])
        
        dataset = Dataset(data=[{'image':input_image}], transform=self.pre_transforms)
        dataloader = DataLoader(dataset, batch_size=1)

        for i in dataloader:
            logging.info(f'Preprocessed input size is {i["image"].shape}')
            o = net(i['image'].to(device))
            logging.info(f'Output size is {o.shape}')
            i['pred'] = o.detach()
            out = [self.post_transforms(x) for x in decollate_batch(i)]
        op_output.set("Output saved",'output_msg')

        
