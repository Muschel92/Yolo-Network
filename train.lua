--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'

--[[
   1. Setup SGD optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay,
    nesterov = true
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
        {  1,     6,   5e-4,   5e-4 },
        { 7,     12,   1.25e-3,   5e-4  },
        { 13,     18,   2.5e-3,   5e-4 },
        { 18,     500,   5e-3,   5e-4 },
        { 501,    750,   5e-4,   5e-4 },
        { 751,    1000,   5e-5,   5e-4 },
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


-- 2. Create loggers.
--trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

local batchNumber
local loss_epoch
local numImages
local ex_boxes = {}
local gt_boxes = {}
local list_ims = {}
local firstImages 

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)
   
   numImages = 0
   ex_boxes = {}
   gt_boxes = {}
   list_ims = {}
   firstImages = true
   
   local params, newRegime, shed = paramsForEpoch(epoch)
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
   
   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()

    e_reg = 0
    e_class = 0
    e_conf = 0
    e_neg = 0
    
   indices = torch.randperm(#image_roidb_train):long():split(opt.batchSize)
   
   if (#image_roidb_train % opt.batchSize ~= 0) then
    indices[#indices] = nil
   end
   
   epochL = opt.epochSize
   
   if opt.epochSize > #indices then
     epochL = #indices
   end
      
   local tm = torch.Timer()
   loss_epoch = 0
   train_reg_accuracy = 0
   
   for t,v in ipairs(indices) do
      
      local roidbs = generate_batch_roidbs(v, image_roidb_train)   
      local im, correction = loadTrainBatch(roidbs)
      
     trainBatch(im, correction)
      --[[donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
           print(image_roidb_train)
            local roidbs = generate_batch_roidbs(v, image_roidb_train)   
            local correction = loadTrainBatch(roidbs)
            return correction
         end,
         -- the end callback (runs in the main thread)
         trainBatch
      )]]--
    
    if t == epochL then
      break
    end
   end 
   donkeys:synchronize()
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
      
      correct_outputs:resize(outputs:size())
      positive_indexes:resize(outputs:size())
      
      -- for every image in batch
      for i =1, #correction do
        numImages = numImages + 1
        outputs[i], correct_outputs[i], positive_indexes[i], temp = calculate_correct_output(outputs[i], correction[i])  
        if numImages <= 12 then
          table.insert(ex_boxes, temp)
          table.insert(gt_boxes, correction[i][2])
          table.insert(list_ims, torch.Tensor(inputs[i]:size()):copy(inputs[i]))
        end
      end
      
      local crit_out = {}
      local crit_corr = {}
      
      -- labels
      --print('labels')
      local labels_out = outputs[positive_indexes:eq(1)]
      labels_out:resize(opt.batchSize, opt.grid_size[1] * opt.grid_size[2] , opt.nClasses)
      table.insert(crit_out, labels_out)
      
      local labels_corr = correct_outputs[positive_indexes:eq(1)]
      labels_corr:resize(opt.batchSize, opt.grid_size[1] * opt.grid_size[2] , opt.nClasses)
      table.insert(crit_corr, labels_corr)
      
      local temp = nn.MSECriterion():cuda()
      err_class = temp:forward(labels_out, labels_corr)
      class_grad = temp:backward(labels_out, labels_corr)
      --print(torch.max(class_grad))
      
      -- regression
      --print('reg')
      local reg_out = outputs[positive_indexes:eq(3)]
      reg_out:resize((reg_out:size(1) / 4), 4)
      table.insert(crit_out, reg_out)
      
      local reg_corr = correct_outputs[positive_indexes:eq(3)]
      --print(reg_corr:size())
      reg_corr:resize((reg_corr:size(1) /  4), 4)
      table.insert(crit_corr, reg_corr)
      --print('Regression')
      err_reg = temp:forward(reg_out, reg_corr)
      grad_reg = temp:backward(reg_out, reg_corr) 
      --print(torch.max(grad_reg))
      
      -- positive confidence
      --print('conf')
      local conf_pos_out = outputs[positive_indexes:eq(4)]
      table.insert(crit_out, conf_pos_out) 
      
      local conf_pos_corr = correct_outputs[positive_indexes:eq(4)]
      table.insert(crit_corr, conf_pos_corr)
      --print('Confidence')
      err_conf = temp:forward(conf_pos_out, conf_pos_corr)
      grad_conf = temp:backward(conf_pos_out, conf_pos_corr)
      --print(torch.max(grad_conf))
      
      --negative confidence
      --print('neg')
      local conf_neg_out = outputs[positive_indexes:eq(2)]
      table.insert(crit_out, conf_neg_out) 
      
      local conf_neg_corr = correct_outputs[positive_indexes:eq(2)]
      table.insert(crit_corr, conf_neg_corr)
      --print('negative confidence')
      err_neg = temp:forward(conf_neg_out, conf_neg_corr)
      grad_neg = temp:backward(conf_neg_out, conf_neg_corr)
      --print(torch.max(grad_neg))
      
      train_reg_accuracy = train_reg_accuracy + torch.mean(conf_pos_corr) / opt.batchSize
      err = criterion:forward(crit_out, crit_corr)
      local gradOut = criterion:backward(crit_out, crit_corr)
      
      -- transform gradOutput back to original size
      gradOutputs:resize(outputs:size()):fill(0)
      
      --print(torch.max(gradOut[1]))
      gradOutputs[positive_indexes:eq(1)] = gradOut[1]
      --print(torch.max(gradOut[2]))
      gradOutputs[positive_indexes:eq(3)]:copy(gradOut[2])
      --print(torch.max(gradOut[3]))
      gradOutputs[positive_indexes:eq(4)] = gradOut[3]
      --print(torch.max(gradOut[4]))
      gradOutputs[positive_indexes:eq(2)] = gradOut[4]
      
      
      
      --print('MAX/MIN GRAD')
      --print(torch.max(gradOutputs))
      --print(torch.min(gradOutputs))
      
      model:backward(inputs, gradOutputs)
      gradParameters:div(opt.batchSize)
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
