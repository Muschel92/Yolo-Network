
function box_size(boxes)
  local box = boxes:clone()
  
  box[{{},3}]:add(-box[{{},1}])
  box[{{},4}]:add(-box[{{},2}])
  
  box[{{},4}]:cmul(box[{{},3}])
  
  return box[{{},4}]
end
