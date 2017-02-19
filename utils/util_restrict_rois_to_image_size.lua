  -- function to restrict a bounding box to the size of the image
  --values less than 1 set to 1
  --values greater than image size set to image height
  --upper value smaller than lower value set to lower value
  

function restrict_rois_to_image_size(ex_boxes, image_size)
  
  ex_boxes[{{},1}][ex_boxes[{{},1}]:lt(1)] = 1  
  ex_boxes[{{},1}][ex_boxes[{{},1}]:gt(image_size[1])] = image_size[1] 
  ex_boxes[{{},3}][ex_boxes[{{},3}]:gt(image_size[1])] = image_size[1]
  ex_boxes[{{},3}][ex_boxes[{{},3}]:lt(ex_boxes[{{},1}])] = ex_boxes[{{},1}][ex_boxes[{{},3}]:lt(ex_boxes[{{},1}])]
  
  ex_boxes[{{},2}][ex_boxes[{{},2}]:lt(1)] = 1
  ex_boxes[{{},2}][ex_boxes[{{},2}]:gt(image_size[2])] = image_size[2]
  ex_boxes[{{},4}][ex_boxes[{{},4}]:gt(image_size[2])] = image_size[2]
  ex_boxes[{{},4}][ex_boxes[{{},4}]:lt(ex_boxes[{{},2}])] = ex_boxes[{{},2}][ex_boxes[{{},4}]:lt(ex_boxes[{{},2}])]
  
  return ex_boxes
end