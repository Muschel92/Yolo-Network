-- function to calculate bounding boxes from the yolo regression values
-- c_x = (reg + grid_x -1) * grid_size
-- height = reg² * image_size
-- x_1/2 = c_x +/- 1/2 height
function calc_box_from_reg(reg, grid_x, grid_y)
  
    local regression = reg:clone()
    local grid_size = torch.cdiv(opt.imageSize, opt.grid_size)
    local boxes = torch.Tensor(grid_x:size(1),4)
    
    -- c_x = reg * grid_size + (grid_x - 1) * grid_size
    regression[{{},1}]:add(grid_x -1):mul(grid_size[1])
    regression[{{},2}]:add(grid_y -1):mul(grid_size[2])
    
    -- height = reg² * image_size
    regression[{{},3}]:pow(2):mul(opt.imageSize[1]):div(2)
    regression[{{},4}]:pow(2):mul(opt.imageSize[2]):div(2)
  
    -- c_x = c_x +/- 1/2 height
    boxes[{{},1}] = regression[{{},1}] - regression[{{},3}]
    boxes[{{},3}] = regression[{{},1}] + regression[{{},3}]
    boxes[{{},2}] = regression[{{},2}] - regression[{{},4}]
    boxes[{{},4}] = regression[{{},2}] + regression[{{},4}]
    
    return torch.round(boxes)
end
