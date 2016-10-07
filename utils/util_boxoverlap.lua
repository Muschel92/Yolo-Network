require('torch')

-- return the intersection in a torch.Tensor(number of boxes, 3)
function boxoverlap (boxes1, boxes2)
  overlap = torch.Tensor(boxes1:size(1), boxes2:size(1))
  
  for i = 1, boxes2:size(1) do
      
    local x1_1 = boxes1:index(2, torch.LongTensor{1})
    local x1_2 = torch.Tensor(x1_1:size(1)):fill(boxes2[i][1])
    
    local x2_1 = boxes1:index(2, torch.LongTensor{3})
    local x2_2 = torch.Tensor(x2_1:size(1)):fill(boxes2[i][3])
    
    local y1_1 = boxes1:index(2, torch.LongTensor{2})
    local y1_2 = torch.Tensor(y1_1:size(1)):fill(boxes2[i][2])
    
    local y2_1 = boxes1:index(2, torch.LongTensor{4})
    local y2_2 = torch.Tensor(y2_1:size(1)):fill(boxes2[i][4])
    
    
    local x1 = x1_1:cat(x1_2, 2)
    local x2 = x2_1:cat(x2_2, 2)
    local y1 = y1_1:cat(y1_2, 2)
    local y2 = y2_1:cat(y2_2, 2)

    local x_min, min_x_idx = torch.max(x1, 2)
    local x_max, max_x_idx = torch.min(x2, 2)
    local y_min, min_y_idx = torch.max(y1,2)
    local y_max, max_y_idx = torch.min(y2,2)
    
    local height = x_max - x_min
    local width = y_max - y_min
    
    local intersection = torch.cmul(height,width)
    
    local area1 = torch.cmul((x2_1 - x1_1),(y2_1 - y1_1))
    local area2 = torch.cmul((x2_2 - x1_2),(y2_2 - y1_2))

    overlap[{{}, i}] = torch.cdiv(intersection, area1 + area2 - intersection)
        
    overlap[{{}, i}][width:le(0)] = 0
    overlap[{{}, i}][height:le(0)] = 0
    
    
    
  end

  return overlap
  
end