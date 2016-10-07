
function prepare_image_roidb(opt, roidbs)

  for i = 1, #roidbs do
        bbox_targets = compute_targets(opt, roidbs[i])
        roidbs[i].bbox_targets = bbox_targets
  end
  
  return roidbs
end

-- computes the grid cell and the regression for x,y, widht, height
function compute_targets(opt, roidb)
  
  local scale = torch.cdiv(opt.imageSize, roidb.size:float())
  
  local rois = scale_rois(roidb.boxes:float(), scale)
  
  local height = rois[{{},1}] - rois[{{},3}]
  local width = rois[{{},2}] - rois[{{},4}]
  
  local c_x = rois[{{},1}] + torch.div(height, 2)
  local c_y = rois[{{},2}] + torch.div(width, 2)
  
  local grid_size = torch.cdiv(opt.imageSize, opt.grid_size)
  
  -- get the grid cell of boxes
  local grid_x = torch.div(c_x, grid_size[1])
  local grid_y = torch.div(c_y, grid_size[2])
  
  local bbox_targets = torch.Tensor(rois:size(1), 7)
  bbox_targets[{{}, 1}] = roidb.labels[{{}}]
  
  bbox_targets[{{}, 2}] = torch.abs(grid_x[{{}}])
  bbox_targets[{{}, 3}] = torch.abs(grid_y[{{}}])
  
  bbox_targets[{{}, 4}] = grid_x - bbox_targets[{{}, 1}]
  bbox_targets[{{}, 5}] = grid_y - bbox_targets[{{}, 2}]
  
  bbox_targets[{{}, 6}] = torch.div(height, opt.imageSize[1])
  bbox_targets[{{}, 7}] = torch.div(width, opt.imageSize[2])
  
  return bbox_targets
  
end
