
--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'image'

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------


local loadSize   = {3, opt.imageSize[1], opt.imageSize[2]}
local sampleSize = {3, opt.cropSize, opt.cropSize}

local indices = torch.randperm(#image_roidb_train):long():split(opt.batchSize)
   
 if (#image_roidb_train % opt.batchSize ~= 0) then
  indices[#indices] = nil
 end
 
 local counter = 1


local function loadImage(path)
   local input = image.load(path, 3, 'float')
   -- find the smaller dimension, and resize it to loadSize (while keeping aspect ratio)  

   input = image.scale(input, loadSize[2], loadSize[3])
      
   input[1] = input[1] -image_mean[1]
   input[2] = input[2] -image_mean[2]
   input[3] = input[3] -image_mean[3]
   
   input = input:index(1, torch.LongTensor{3,2,1})
  
   return input
end

function random_crop(im_path, size, new_size,  boxes)
    im = image.load(im_path, 3, 'float')
    -- calculate crop
    local crop_x = torch.rand(1) * 0.2 * size[1] 
    local crop_y = torch.rand(1) * 0.2 * size[2] 
    
    -- calculate crop size
    local crop_size = torch.Tensor(2)
    crop_size[1] = size[1] - crop_x[1]
    crop_size[2] = size[2] - crop_y[1]
    crop_size:round()
    
    -- calculate crop coordinates
    local x1 = torch.rand(1) * crop_x[1]
    local y1 = torch.rand(1) * crop_y[1]
    
    x1:floor()
    y1:floor()
    
    local x2 = x1 + crop_size[1]
    local y2 = y1 + crop_size[2]
    
    -- crop image
    im = image.crop(im, y1[1], x1[1], y2[1], x2[1])
    -- scale image
    im = image.scale(im, new_size[2], new_size[1])
    
    local overlap1 = box_size(boxes)
    
    -- crop boxes
    boxes[{{},1}]:add(-x1[1])
    boxes[{{},2}]:add(-y1[1])
    boxes[{{},3}]:add(-x1[1])
    boxes[{{},4}]:add(-y1[1])
    
    boxes = restrict_rois_to_image_size(boxes, crop_size) 
    
    local overlap2 = box_size(boxes)
    
    -- extract boxes which have overlap > 0.6
    local prop = overlap1:cdiv(overlap2)
    local keep_idx = binaryToIdx(prop:gt(0.6))
    boxes = boxes:index(1, keep_idx)
    
    -- scale boxes to new size
    local scale = torch.cdiv(new_size, crop_size)
    boxes = scale_rois(boxes, scale)
    
    -- normalize image
    im[1] = im[1] -image_mean[1]
    im[2] = im[2] -image_mean[2]
    im[3] = im[3] -image_mean[3]

    im = im:index(1, torch.LongTensor{3,2,1})
    
    return im, boxes
  
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
--[[
   Section 1: Create a train data loader (trainLoader),
   which does class-balanced sampling from the dataset and does a random crop
--]]

-- function to load the image, jitter it appropriately (random crops etc.)
function loadTrainBatch()

    if counter > #indices then
      indices = torch.randperm(#image_roidb_train):long():split(opt.batchSize)
   
      if (#image_roidb_train % opt.batchSize ~= 0) then
        indices[#indices] = nil
      end

      counter = 1
    end
    
    -- get next batch roidbs
    local roidbs = generate_batch_roidbs(indices[counter], image_roidb_train) 
    
    -- calculate next batch    
    local batch = torch.Tensor(#roidbs, 3, opt.imageSize[1], opt.imageSize[2])
    local bbox_targets = {}
    
    for i = 1,#roidbs do
      local img, correction = trainHook(roidbs[i])
      table.insert(bbox_targets, correction)
      batch[i] = img
    end
    counter = counter + 1
    
    return batch, bbox_targets
end

function trainHook (roidb)
   collectgarbage()
   
   local out, rois = random_crop(roidb.path, roidb.size:clone(), opt.imageSize:clone(), roidb.boxes:clone())


   -- index for all positive boxes 
   local pos_boxes = torch.Tensor(opt.grid_size[1], opt.grid_size[2], 2):fill(0)
   local labels = torch.Tensor(opt.grid_size[1], opt.grid_size[2], opt.nClasses):fill(0)
   
   -- do hflip with probability 0.5
    if torch.uniform() > 0.5 then 
      out = image.hflip(out) 
      rois = flip_rois(rois, opt.imageSize)
    end
    
    -- calculate height and width
    local height = rois[{{},3}] - rois[{{},1}]
    local width = rois[{{},4}] - rois[{{},2}]

    -- calculate center point
    local c_x = rois[{{},1}] + torch.div(height, 2)
    local c_y = rois[{{},2}] + torch.div(width, 2)

    local grid_size = torch.cdiv(opt.imageSize, opt.grid_size)

    -- get the grid cell of boxes
    local grid_x = torch.div(c_x, grid_size[1])
    local grid_y = torch.div(c_y, grid_size[2])

    -- calculate the bbox targets for every gt_box
    local bbox_targets = torch.Tensor(rois:size(1), 5)
    
    -- label
    bbox_targets[{{}, 1}]:copy(roidb.labels)

    -- grid offset to center
    bbox_targets[{{}, 2}] = (grid_x - torch.floor(grid_x))
    bbox_targets[{{}, 3}] = (grid_y - torch.floor(grid_y))

    -- height and width normalized to image size and sqrt(w/h)
    bbox_targets[{{}, 4}] = torch.div(height, opt.imageSize[1]):sqrt()
    bbox_targets[{{}, 5}] = torch.div(width, opt.imageSize[2]):sqrt()
    
    grid_x:ceil()
    grid_y:ceil()
    
    -- mark all the positive boxes / one box possible for every cell
    for i = 1, bbox_targets:size(1) do
      if pos_boxes[grid_x[i]][grid_y[i]][1] ~= 0 then
        pos_boxes[grid_x[i]][grid_y[i]][2] = i
      else
        pos_boxes[grid_x[i]][grid_y[i]][1] = i
      end           
      labels[grid_x[i]][grid_y[i]]:fill(2)
      labels[grid_x[i]][grid_y[i]][bbox_targets[{i, 1}]] = 1
      
    end
    
   
   local correction = {}
   
   -- 1) the bbox_targets
   table.insert(correction, bbox_targets)
   -- 2) the actual rois
   table.insert(correction, rois)
   -- 3) the position of the positive boxes
   table.insert(correction, pos_boxes)
   -- 4) the labels and their position
   table.insert(correction, labels)
   -- 5) The grid_x of pos_box i
   table.insert(correction, grid_x)
   -- 6) The grid_y of pos_box i
   table.insert(correction, grid_y)
   
   return out, correction
end

collectgarbage()


-- End of train loader section
--------------------------------------------------------------------------------
--[[
   Section 2: Create a test data loader (testLoader),
   which can iterate over the test set and returns an image's
--]]

-- function to load the image
function loadValBatch(roidbs)
  local batch = torch.Tensor(#roidbs, 3, opt.imageSize[1], opt.imageSize[2])
  local bbox_targets = {}
  for i = 1,#roidbs do
    local img, correction = valHook(roidbs[i])
    table.insert(bbox_targets, correction)
    batch[i] = img
  end
  
  return batch, bbox_targets
  
end


valHook = function(roidb)
   collectgarbage()
   local input = loadImage(roidb.path)
   
   local scale = torch.cdiv(opt.imageSize, roidb.size)
  
  local rois = scale_rois(roidb.boxes, scale)
  
  local height = rois[{{},1}] - rois[{{},3}]
  local width = rois[{{},2}] - rois[{{},4}]
  
  local c_x = rois[{{},1}] + torch.div(height, 2)
  local c_y = rois[{{},2}] + torch.div(width, 2)
  
  local grid_size = torch.cdiv(target_size, opt.grid_size)
  
  -- get the grid cell of boxes
  local grid_x = torch.div(c_x, grid_size[1])
  local grid_y = torch.div(c_y, grid_size[2])
  
  local bbox_targets = torch.Tensor(rois:size(1), 7)
  bbox_targets[{{}, 1}] = roidb.labels[{{}}]
  
  bbox_targets[{{}, 2}] = torch.abs(grid_x[{{}}])
  bbox_targets[{{}, 3}] = torch.abs(grid_y[{{}}])
  
  bbox_targets[{{}, 4}] = grid_x - bbox_targets[{{}, 2}]
  bbox_targets[{{}, 5}] = grid_y - bbox_targets[{{}, 3}]
  
  bbox_targets[{{}, 6}] = torch.div(height, opt.imageSize[1])
  bbox_targets[{{}, 7}] = torch.div(width, opt.imageSize[2])
   
   -- mean/std
   for i=1,3 do -- channels
      if mean then out[{{i},{},{}}]:add(-mean[i]) end
      if std then out[{{i},{},{}}]:div(std[i]) end
   end
   return out, bbox_targets
end

