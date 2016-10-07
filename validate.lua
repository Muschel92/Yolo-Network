--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime

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

-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local targets = torch.CudaTensor()
local correct_outputs = torch.CudaTensor()
local positive_indexes = torch.CudaTensor()
-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function validate()
  
  val_false_pos = 0
  val_false_neg = 0 
  val_corr_class = 0
  val_class = 0
  val_reg = 0
  val_conf = 0
  val_neg = 0
  
   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:evaluate()

   indices = torch.randperm(#image_roidb_val):long():split(opt.batchSize)
   
   if (#image_roidb_val % opt.batchSize ~= 0) then
    indices[#indices] = nil
   end
   
   epochL = opt.epochSize
   
   if opt.epochSize > #indices then
     epochL = #indices
   end
      
   local tm = torch.Timer()
   loss_epoch = 0
   
   for t,v in ipairs(indices) do
      local roidbs = generate_batch_roidbs(v, image_roidb_train)   
      local im, correction = loadTrainBatch(roidbs)
      
      inputs:resize(im:size()):copy(im)
      outputs = model:forward(inputs)
      
      correct_outputs:resize(outputs:size())
      positive_indexes:resize(outputs:size())
      
      -- for every image in batch
      for i =1, #correction do
        outputs[i], correct_outputs[i], positive_indexes[i] = calculate_correct_output(outputs[i], correction[i]) 
        local p_table = test_output(outputs[i], correction[i])
        
        val_false_pos = val_false_pos + p_table[1]
        val_false_neg = val_false_neg + p_table[2]
        val_corr_class = val_corr_class + p_table[3]
        
      end
      
      val_false_pos = val_false_pos / #correction
      val_false_neg = val_false_neg / #correction
      val_corr_class = val_corr_class / #correction
      
      local crit_out = {}
      local crit_corr = {}
      
      local temp = nn.MSECriterion()
      -- labels
      local labels_out = outputs[positive_indexes:eq(1)]
      labels_out:resize(opt.batchSize, opt.grid_size[1] * opt.grid_size[2] , opt.nClasses)
      table.insert(crit_out, labels_out)
      
      local labels_corr = correct_outputs[positive_indexes:eq(1)]
      labels_corr:resize(opt.batchSize, opt.grid_size[1] * opt.grid_size[2] , opt.nClasses)
      table.insert(crit_corr, labels_corr)
      val_class = val_class + temp:forward(labels_out, labels_corr)
      
      -- regression
      local reg_out = outputs[positive_indexes:eq(3)]
      reg_out:resize(opt.batchSize, (reg_out:size(1) / opt.batchSize) / 4, 4)
      table.insert(crit_out, reg_out)
      
      local reg_corr = correct_outputs[positive_indexes:eq(3)]
      reg_corr:resize(opt.batchSize, (reg_corr:size(1) / opt.batchSize) / 4, 4)
      table.insert(crit_corr, reg_corr)
      val_reg = val_reg + temp:forward(reg_out, reg_corr)
      
      -- positive confidence
      local conf_pos_out = outputs[positive_indexes:eq(4)]
      conf_pos_out:resize(opt.batchSize, (conf_pos_out:size(1) / opt.batchSize), 1)
      table.insert(crit_out, conf_pos_out) 
      
      local conf_pos_corr = correct_outputs[positive_indexes:eq(4)]
      conf_pos_corr:resize(opt.batchSize, (conf_pos_corr:size(1) / opt.batchSize), 1)
      table.insert(crit_corr, conf_pos_corr)
      val_conf = val_conf + temp:forward(conf_pos_out, conf_pos_corr)
      
      --negative confidence
      local conf_neg_out = outputs[positive_indexes:eq(2)]
      conf_neg_out:resize(opt.batchSize, (conf_neg_out:size(1) / opt.batchSize), 1)
      table.insert(crit_out, conf_neg_out) 
      
      local conf_neg_corr = correct_outputs[positive_indexes:eq(2)]
      conf_neg_corr:resize(opt.batchSize, (conf_neg_corr:size(1) / opt.batchSize), 1)
      table.insert(crit_corr, conf_neg_corr)
      val_neg = val_neg + temp:forward(conf_neg_out, conf_neg_corr)
      
      local err = criterion:forward(crit_out, crit_corr)
      
      
      loss_epoch = loss_epoch + err
      val_reg_accuracy = val_reg_accuracy + torch.mean(conf_pos_corr)
   end 
   donkeys:synchronize()
   cutorch.synchronize()

   val_loss = loss_epoch / epochL
   val_reg_accuracy = val_reg_accuracy / epochL
   val_reg = val_reg /epochL
   val_class = val_class /epochL
   val_conf = val_conf / epochL
   val_neg = val_neg / epochL

   print(string.format('Epoch: [%d][VALIDATION SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t ',
                       epoch, tm:time().real, val_loss))
   print('\n')

   -- save model
   collectgarbage()

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   model:clearState()
   saveDataParallel(paths.concat(opt.save .. 'Models/', 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
   torch.save(paths.concat(opt.save .. 'Models/', 'optimState_' .. epoch .. '.t7'), optimState)
end -- of train()
-------------------------------------------------------------------------------------------
