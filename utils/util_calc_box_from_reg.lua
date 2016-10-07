
local boxes = torch.CudaTensor()

function calc_box_from_reg(regression, grid_x, grid_y)
  
    local grid_size = torch.cdiv(opt.imageSize, opt.grid_size)
    boxes:resize(grid_x:size(1),4)
    
    local ctr_x = torch.add(grid_x ,torch.mul(regression[{{},1}], grid_size[1]))
    local ctr_y = torch.add(grid_y ,torch.mul(regression[{{},2}], grid_size[2]))
    
    boxes[{{},1}] = ctr_x - torch.div(torch.mul(regression[{{},3}], opt.imageSize[1]),2)
    boxes[{{},3}] = ctr_x + torch.div(torch.mul(regression[{{},3}], opt.imageSize[1]),2)
    boxes[{{},2}] = ctr_x - torch.div(torch.mul(regression[{{},4}], opt.imageSize[2]),2)
    boxes[{{},4}] = ctr_x + torch.div(torch.mul(regression[{{},4}], opt.imageSize[2]),2)
    
    return boxes
end
