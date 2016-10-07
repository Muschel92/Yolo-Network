
-- scales a bbox according to a new scale of an image
-- image scale (height width) *(scale_h scale_w) = (new_height new_widht)
function scale_rois(rois, im_scale)
  scaled_rois = torch.Tensor(rois:size(1), 4)
  
  scaled_rois[{{},1}] = torch.round(torch.mul(rois[{{},1}], im_scale[1]))
  scaled_rois[{{},2}] = torch.round(torch.mul(rois[{{},2}], im_scale[2]))  
  scaled_rois[{{},3}] = torch.round(torch.mul(rois[{{},3}], im_scale[1]))
  scaled_rois[{{},4}] = torch.round(torch.mul(rois[{{},4}], im_scale[2]))

  return (scaled_rois)
end