local width_idx = torch.range(0, opt.boxes_per_grid - 1):long() * 5 + 4
local height_idx = torch.range(0, opt.boxes_per_grid - 1):long() * 5 + 3
local x_idx = torch.range(0, opt.boxes_per_grid - 1):long() * 5 + 1
local y_idx = torch.range(0, opt.boxes_per_grid - 1):long() * 5 + 2
local conf_idx = torch.range(0, opt.boxes_per_grid - 1):long() * 5 + 5
local class_idx = torch.range(opt.boxes_per_grid * 5 +1, opt.boxes_per_grid * 5 + 20):long()

-- return pos_rois with indexes as follows
-- 1: for positive labels 
-- 2: for non-object confidences which should be zero
-- 3: for positive correct regression of grid
-- 4: for positive confidence

-- returns modified output in which all values are set to zero if needed

-- return the correct_output which the output can be compared to



function calculate_correct_output(output, correction)
  
  -- set class output to zero for non-object positions
  output[{{},{}, {opt.boxes_per_grid *5 + 1, output:size(3)}}][torch.eq(correction[4], 0)] = 0

  -- indexes for positive output
  local pos_rois = torch.Tensor(output:size()):fill(0)
  
  -- set index for positive objects to 1
  pos_rois[{{},{},{opt.boxes_per_grid*5 +1, output:size(3)}}] = 1
  -- set index for confidence to 2
  pos_rois:indexFill(3,conf_idx, 2)
  
  -- set the non-correct classes of correction to zero
  correction[4][correction[4]:eq(2)] = 0
  
  -- correct output
  local correct_output = torch.Tensor(output:size()):fill(0)
  -- set positive labels
  correct_output[{{},{},{opt.boxes_per_grid*5 +1, output:size(3)}}] = correction[4]
  --[torch.eq(correction[4], 1)] = 1
       
  -- get bbox regression  
  -- x coordinates
  local x = output:index(3, x_idx)
  -- y coordinates
  local y = output:index(3, y_idx)
  -- width coordinates
  local width = output:index(3, width_idx)
  -- height coordinates
  local height = output:index(3, height_idx)
  
  -- get nr of pos boxes plus the gt box for every pos box
  local nr_of_pos_boxes = torch.sum(correction[3]:gt(0))
  -- nr of positive positions
  local nr_of_pos_positions = torch.sum(correction[3][{{},{},1}]:gt(0))
  -- positions with multiple boxes
  local double_boxes = correction[3][{{},{},1}]:gt(0):float():cmul(correction[3][{{},{},2}]:gt(0):float())
  -- tensor of all position of boxes, 1's for two boxes in the same grid cell
  double_boxes = double_boxes[correction[3][{{},{},1}]:gt(0)]
  
  local reg_index = torch.LongTensor(nr_of_pos_boxes)
  local index = 1
  
  -- get an order of the boxes according to their spatial position
  for i = 1, nr_of_pos_positions do
    reg_index[index] = i
    if double_boxes[i] > 0 then
      reg_index[index + 1] = i
      index = index + 1
    end
    index  = index + 1
  end
  
  -- determine the gt box for every positive box
  local pos_box_idx = correction[3][correction[3]:gt(0)]
  pos_box_idx:resize(nr_of_pos_boxes)
  
  -- the regression of the grid cells which contain an object
  local reg = torch.zeros(nr_of_pos_boxes * opt.boxes_per_grid, 4)
  
  -- the actual boxes of the grid cells
  local boxes= torch.zeros(nr_of_pos_boxes * opt.boxes_per_grid, 4)
  
  
  for i = 1, opt.boxes_per_grid do
    
    -- index of i'th box per grid_cell
    local range = torch.range(0, nr_of_pos_boxes -1):long() * opt.boxes_per_grid + i
  
    -- get x, y, width, height of the boxes i of positive grid cells
    local x_temp = x[{{},{},i}][correction[3][{{},{},1}]:gt(0)]
    x_temp:resize(nr_of_pos_positions)
    local y_temp = y[{{},{},i}][correction[3][{{},{},1}]:gt(0)]
    y_temp:resize(nr_of_pos_positions)
    
    -- get predicts sqrt of width and height
    local height_temp = height[{{},{},i}][correction[3][{{},{},1}]:gt(0)]
    height_temp:resize(nr_of_pos_positions)
    local width_temp = width[{{},{},i}][correction[3][{{},{},1}]:gt(0)]
    width_temp:resize(nr_of_pos_positions)
    
    -- concat the regression values
    local reg_temp = x_temp:cat(y_temp, 2):cat(height_temp, 2):cat(width_temp,2)
    
    -- fill the global regression with the values
    reg:indexCopy(1, range, reg_temp:index(1, reg_index))
    
    -- calculate the boxes from regression output
    local box_temp = calc_box_from_reg(reg:index(1, range), correction[5]:index(1, pos_box_idx:long()), correction[6]:index(1, pos_box_idx:long()))
    boxes:indexCopy(1, range, box_temp)
    --boxes[index_reg:gt(0)] = box_temp:float()
  end
  
  boxes = restrict_rois_to_image_size(boxes, opt.imageSize) 
  
  -- calculate the overlap between estimated boxes and ground truth
  local overlap = boxoverlap(boxes, correction[2])
  local ex_boxes = boxes:clone()
  
  -- Split boxes to boxes for every positive box
  boxes = boxes:split(opt.boxes_per_grid, 1)
  --reg = reg:split(opt.boxes_per_grid, 1)
  
  local box_index = torch.LongTensor(nr_of_pos_boxes)
  index = 1
  
  for i = 1, nr_of_pos_positions do
    
      -- check if current position contains multiple boxes
      if double_boxes[i] > 0 then      
        -- get the box with greatest overlap
        local o1, idx1 = torch.max(overlap[{{(index-1) *opt.boxes_per_grid +1, index *opt.boxes_per_grid},{pos_box_idx[index]}}], 1)
        idx1:resize(idx1:numel())
        o1:resize(o1:numel())
      
        local o2, idx2 = torch.max(overlap[{{(index) *opt.boxes_per_grid +1, (index + 1) *opt.boxes_per_grid},{pos_box_idx[index + 1]}}], 1)
        idx2:resize(idx2:numel())
        o2:resize(o2:numel())
        
        -- assign box with highest overlap if both have max overlap with same box
        if idx1[1] == idx2[1] then
          if o1[1] > o2[1] then
            if idx1[1] == 1 then
              idx2[1] = 2
            else
              idx2[1] = 1
            end
          else 
            if idx2[1] == 1 then
              idx1[1] = 2
            else
              idx1[1] = 1
            end
          end
        end
        
        box_index[index] = (index -1) * opt.boxes_per_grid + idx1[1]
        box_index[index +1 ] = (index) * opt.boxes_per_grid + idx2[1]
        
        -- set x,y,w,h, of box as positive
        pos_rois[{correction[5][pos_box_idx[index]], correction[6][pos_box_idx[index]], {(idx1[1] -1) * 5 + 1, idx1[1] * 5 -1}}]:fill(3)
        pos_rois[{correction[5][pos_box_idx[index + 1]], correction[6][pos_box_idx[index+1]], {(idx2[1] -1) * 5 + 1, idx2[1] * 5 -1}}]:fill(3)
    
        -- set c of box positive
        pos_rois[{correction[5][pos_box_idx[index]], correction[6][pos_box_idx[index]], idx1[1] * 5 }] = 4
        pos_rois[{correction[5][pos_box_idx[index + 1]], correction[6][pos_box_idx[index + 1]], idx2[1] * 5 }] = 4
    
        -- copy the target regression into correct output
        correct_output[{correction[5][pos_box_idx[index]], correction[6][pos_box_idx[index]], {(idx1[1] -1) * 5 + 1, idx1[1] * 5 -1}}]:copy(correction[1][pos_box_idx[index]][{{2,5}}])
        correct_output[{correction[5][pos_box_idx[index + 1]], correction[6][pos_box_idx[index + 1]], {(idx2[1] -1) * 5 + 1, idx2[1] * 5 -1}}]:copy(correction[1][pos_box_idx[index + 1]][{{2,5}}])
      
        -- copy the correct confidence as the IoU
        correct_output[{correction[5][pos_box_idx[index]], correction[6][pos_box_idx[index]], idx1[1] *5}] = o1[1]
        correct_output[{correction[5][pos_box_idx[index + 1]], correction[6][pos_box_idx[index + 1]], idx2[1] *5}] = o2[1]
      
        index = index +2
      
      else
      
        local o, idx = torch.max(overlap[{{(i-1) *opt.boxes_per_grid +1, i *opt.boxes_per_grid},{pos_box_idx[i]}}], 1)
        idx:resize(idx:numel())
        o:resize(o:numel())
        
        -- set x,y,w,h, of box as positive
        pos_rois[{correction[5][pos_box_idx[i]], correction[6][pos_box_idx[i]], {(idx[1] -1) * 5 + 1, idx[1] * 5 -1}}]:fill(3)
      
        -- set c of box positive
        pos_rois[{correction[5][pos_box_idx[i]], correction[6][pos_box_idx[i]], idx[1] * 5 }] = 4
      
        -- copy the target regression into correct output
        correct_output[{correction[5][pos_box_idx[i]], correction[6][pos_box_idx[i]], {(idx[1] -1) * 5 + 1, idx[1] * 5 -1}}]:copy(correction[1][pos_box_idx[i]][{{2,5}}])
      
        -- copy the correct confidence as the IoU
        correct_output[{correction[5][pos_box_idx[i]], correction[6][pos_box_idx[i]], idx[1] *5}] = o[1]
        
        box_index[index] = (index -1) * opt.boxes_per_grid + idx[1]
        
        index = index +1
      end
  end
  -- set the output to zero if its not needed
  output[torch.eq(pos_rois,0)] = 0
  ex_boxes = ex_boxes:index(1, box_index)
    
  return output, correct_output, pos_rois, ex_boxes
end