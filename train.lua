--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'


-- Setup a reused optimization state (for sgd). If needed, reload it from disk
optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay,
    nesterov = true,
    evalCounter = 0
}


-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- ssexact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local function paramsForEpoch(epoch)
    if opt.LR ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     3000,   1e-3,   5e-4 },
        { 3001,     100000,   1e-2,   5e-4  },
        { 100001,     140000,   1e-3,   5e-4  },
        { 290190,     1000000,   1e-4,   5e-4  },
        { 306310,     1000000,   1e-4,   5e-4 },
       -- { 4,     4,   5e-3,   5e-4 },
        --{ ,     500,   1e-2,   5e-4 },
        { 501,    750,   1e-3,   5e-4 },
        { 751,    1000,   1e-4,   5e-4 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1], regimes
        end
    end
end

local function generate_batch_roidbs(files, images)
  
  local ims = {}
  
  for i = 1,files:size(1) do
    table.insert(ims, images[files[i]])
  end
  
  return ims
  
end 


local batchNumber
local loss_epoch
local numImages
local ex_boxes = {}
local gt_boxes = {}
local list_ims = {}
local firstImages 

-- 3. train - this function handles the high-level training loop,
function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)
   
   numImages = 0
   ex_boxes = {}
   gt_boxes = {}
   list_ims = {}
   firstImages = true

   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()

    e_reg = 0
    e_class = 0
    e_conf = 0
    e_neg = 0
   
   epochL = opt.epochSize
      
   local tm = torch.Timer()
   loss_epoch = 0
   train_reg_accuracy = 0
   
   for i = 1, epochL do
            
    local t = optimState.evalCounter 
    if t == 0 then
      t = 1
    end
    
    local params, newRegime, shed = paramsForEpoch(t)
    learning_rate_shedule = shed
   
    if newRegime then
      optimState = {
         learningRate = params.learningRate,
         learningRateDecay = 0.0,
         momentum = opt.momentum,
         dampening = 0.0,
         weightDecay = params.weightDecay
      }
    end
    print(optimState.learningRate)
    
    local im, correction = loadTrainBatch()     
    trainBatch(im, correction)
    
   end 
   
   --donkeys:synchronize()
   cutorch.synchronize()

   train_loss = loss_epoch / epochL
   train_reg_accuracy = train_reg_accuracy / epochL
   e_class = e_class / epochL
   e_reg = e_reg / epochL
   e_conf = e_conf / epochL
   e_neg = e_neg / epochL

   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t ',
                       epoch, tm:time().real, train_loss))
   print(('Class Err %.4f Reg Err %.4f Conf Err %.4f  Neg Err %.4f'):format( e_class, e_reg, e_conf, e_neg)) 
   print('\n')

   -- save model
   collectgarbage()

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   model:clearState()
   
   if epoch % opt.save_step == 0 then
      saveDataParallel(paths.concat(opt.save .. 'Models/', 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
      torch.save(paths.concat(opt.save .. 'Models/', 'optimState_' .. epoch .. '.t7'), optimState)
    end
end -- of train()
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local targets = torch.CudaTensor()
local correct_outputs = torch.CudaTensor()
local positive_indexes = torch.CudaTensor()
local labels_output = torch.CudaTensor()
local gradOutputs = torch.CudaTensor()
local err_reg = 0
local err_class = 0
local err_conf = 0
local err_neg = 0


local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(imageCPU, correction)
   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   timer:reset()

   -- transfer over to GPU
   inputs:resize(imageCPU:size()):copy(imageCPU)

   local err, outputs
   
   feval = function(x)
      model:zeroGradParameters()
      outputs = model:forward(inputs)
      
      --print('Outputs')
      --print(torch.sum(outputs:ne(outputs)))
      
      -- resize correction container to output size
      correct_outputs:resize(outputs:size()):fill(0)
      positive_indexes:resize(outputs:size()):fill(0)
      labels_output:resize(outputs:size(1), outputs:size(2), outputs:size(3)):fill(0)
      
      -- for every image in batch
      for i =1, #correction do
        numImages = numImages + 1
        local temp1, temp2, temp3, temp4 = calculate_correct_output(outputs[i]:float(), correction[i]) 
        --print(torch.sum(temp3:eq(3)))
        outputs[i]:copy(temp1:cuda())
        correct_outputs[i]:copy(temp2:cuda())
        positive_indexes[i]:copy(temp3:cuda())
        --labels_output[i]:copy(correction[i][7]:cuda())

        --print(torch.sum(positive_indexes:eq(3)))
        
        if numImages <= 12 then
          table.insert(ex_boxes, temp4)
          table.insert(gt_boxes, correction[i][2])
          table.insert(list_ims, torch.Tensor(inputs[i]:size()):copy(inputs[i]))
        end
        
      end
      
      --print(torch.sum(positive_indexes:eq(3)))
      local crit_out = {}
      local crit_corr = {}
      
      -- labels
      local labels_out = outputs[positive_indexes:eq(1)]
      labels_out:resize(opt.batchSize, opt.grid_size[1] , opt.grid_size[2] , opt.nClasses)
      --print(labels_out:size())
      --labels_out = labels_out:permute(1,4,2,3):contiguous()
      table.insert(crit_out, labels_out)      
      local labels_corr = correct_outputs[positive_indexes:eq(1)]
      labels_corr:resize(opt.batchSize, opt.grid_size[1] * opt.grid_size[2] , opt.nClasses)
      table.insert(crit_corr, labels_corr)
      e_class = crit1:forward(labels_out, labels_corr)
      
      -- regression
      local reg_out = outputs[positive_indexes:eq(3)]
      reg_out:resize((reg_out:size(1) / 4), 4)
      table.insert(crit_out, reg_out)      
      local reg_corr = correct_outputs[positive_indexes:eq(3)]
      reg_corr:resize((reg_corr:size(1) /  4), 4)
      table.insert(crit_corr, reg_corr)
      e_reg = crit2:forward(reg_out, reg_corr)
      
      --print(reg_out:cat(reg_corr,2))
      
      -- positive confidence
      local conf_pos_out = outputs[positive_indexes:eq(4)]
      table.insert(crit_out, conf_pos_out)       
      local conf_pos_corr = correct_outputs[positive_indexes:eq(4)]
      table.insert(crit_corr, conf_pos_corr)
      e_conf = crit3:forward(conf_pos_out, conf_pos_corr)

      -- negative confidence
      local conf_neg_out = outputs[positive_indexes:eq(2)]
      table.insert(crit_out, conf_neg_out)      
      local conf_neg_corr = correct_outputs[positive_indexes:eq(2)]
      table.insert(crit_corr, conf_neg_corr)
      e_neg = crit4:forward(conf_neg_out, conf_neg_corr)
      
      train_reg_accuracy = train_reg_accuracy + torch.mean(conf_pos_corr)
      err = criterion:forward(crit_out, crit_corr)
      local gradOut = criterion:backward(crit_out, crit_corr)
      
      -- transform gradOutput back to original size
      gradOutputs:resize(outputs:size()):fill(0)
      
      --print(positive_indexes:size())
      --print(gradOutputs:size())
      --print(gradOut)
      --print(torch.sum(positive_indexes:eq(1)))
      --print(torch.sum(positive_indexes:eq(2)))
      --print(torch.sum(positive_indexes:eq(3)))
      --print(torch.sum(positive_indexes:eq(4)))
      
      gradOutputs[positive_indexes:eq(1)] = gradOut[1]
      gradOutputs[positive_indexes:eq(3)] = gradOut[2]
      gradOutputs[positive_indexes:eq(4)] = gradOut[3]
      gradOutputs[positive_indexes:eq(2)] = gradOut[4]
      
      model:backward(inputs, gradOutputs)
      
      return err, gradParameters
   end
   optim.sgd(feval, parameters, optimState)

   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err
   e_reg = e_reg + err_reg
   e_class = e_class + err_class
   e_conf = e_conf + err_conf
   e_neg = e_neg + err_neg

   -- Calculate top-1 error, and print information
   print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f LR %.0e DataLoadingTime %.3f'):format( epoch, batchNumber, epochL, timer:time().real, err, optimState.learningRate, dataLoadingTime))
   --print(('e_reg: %.4f e_class: %.4f e_conf: %.4f e_neg: %.4f'):format(e_reg, e_class, e_conf, e_neg ))
   
   if numImages >= 12 and firstImages then
    for i = 1,12 do
      --calculate back to original image (bgr->bgr and mean/std calculation)
      
      -- change back from brg to rgb
      list_ims[i] = list_ims[i]:index(1, torch.LongTensor{3,2,1})
         
      -- add mean to image
      list_ims[i] = img_from_mean(list_ims[i], image_mean)
      
      list_ims[i]:mul(255)
      local im_size = torch.Tensor{ list_ims[i]:size(2),  list_ims[i]:size(3)}
      local gt = gt_boxes[i]
      local pos_ex_boxes = ex_boxes[i]
      
      --print(pos_ex_boxes:size())
      -- draw all gt boxes into image
      for j = 1,gt:size(1) do
         list_ims[i] = image.drawRect( list_ims[i]:byte(), gt[{j,2}], gt[{j,1}], gt[{j,4}], gt[{j,3}], {lineWidth = 1, color = {0, 255, 0}})    
      end     
      
      image.save( opt.save .. '/Images/trainGt' .. i .. '.png',  list_ims[i])
      
      -- draw all positive boxes into image
      for j = 1,pos_ex_boxes:size(1) do
        local x2, y2 = 0
        local col = torch.Tensor(3)
        col[1] = torch.random(1,255)
        col[2] = torch.random(1,255)
        col[3] = torch.random(1,255)
        if(pos_ex_boxes[{j,1}] < im_size[1] and pos_ex_boxes[{j,2}] < im_size[2] and pos_ex_boxes[{j,1}] > 0 and pos_ex_boxes[{j,2}] > 0) then
          if (pos_ex_boxes[{j,3}] > im_size[2]) then
            x2 = im_size[1]
          else
            x2 = pos_ex_boxes[{j,3}]
          end
          
          if (pos_ex_boxes[{j,4}] > im_size[1]) then
            y2 = im_size[2]
          else
            y2 = pos_ex_boxes[{j,4}]
          end
          list_ims[i] = image.drawRect(list_ims[i]:byte(), pos_ex_boxes[{j,2}], pos_ex_boxes[{j,1}],pos_ex_boxes[{j,4}], pos_ex_boxes[{j,3}], {lineWidth = 1, color = col})    
          --draw_rect(im, pos_ex_boxes[{j,1}], pos_ex_boxes[{j,2}], x2, y2, {255, 0, 0}) 
        end
      end
            
      image.save(opt.save .. '/Images/trainEx' .. i .. '.png', list_ims[i])
      
    end
    
    firstImages = false
  end
end
