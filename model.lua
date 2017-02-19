
--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

--[[
   1. Create Model
   2. Create Criterion
   3. Convert model to CUDA
]]--

-- 1. Create Network
-- 1.1 If preloading option is set, preload weights from existing models appropriately
if opt.retrain ~= 'none' then
   assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
   print('Loading model from file: ' .. opt.retrain);
   model = loadDataParallel(opt.retrain, opt.nGPU) -- defined in util.lua
elseif opt.new_model then
   paths.dofile('model/' .. opt.netType .. '.lua')
   print('=> Creating model from file: model/' .. opt.netType .. '.lua')
   model = makeDataParallel(model, opt.nGPU)
   torch.save(opt.netPath .. opt.netType ..'_' .. 'yolo' .. '.t7', model)
   model:cuda()
else
   --paths.dofile('models/' .. opt.netType .. '.lua')
   print('=> Loading model')
   print(opt.netPath .. opt.netType .. '.t7')
   model = torch.load(opt.netPath .. opt.netType .. '.t7')
   model:cuda()
   model = makeDataParallel(model, opt.nGPU)
   -- for the model creation code, check the models/ folder
   --if opt.backend == 'cudnn' then
      --require 'cudnn'
      --cudnn.convert(model, cudnn)
   --elseif opt.backend ~= 'nn' then
      --error'Unsupported backend'
   --end
end

-- 2. Create Criterion
criterion = nn.ParallelCriterion()

--Labels
if opt.crossEntropy then
  local weights = torch.Tensor(opt.nClasses):fill(1)
  crit1 = cudnn.SpatialCrossEntropyCriterion(weights)
else
  crit1 = nn.MSECriterion()
end 
--crit1 = nn.MSECriterion()
--crit1.sizeAverage = false
criterion:add(crit1, opt.class_loss)

--Regression
--crit2 = nn.SmoothL1Criterion()
if opt.smoothL1 then
  crit2 = nn.SmoothL1Criterion()
else
  crit2 = nn.MSECriterion()
end 
--crit2.sizeAverage = false
criterion:add(crit2, opt.reg_loss)

-- Positive Confidence
crit3 = nn.MSECriterion()
--crit3.sizeAverage = false
criterion:add(crit3, opt.conf_loss)

--Negative Confidence
crit4 = nn.MSECriterion()
--crit4.sizeAverage = false
criterion:add(crit4, opt.neg_loss)

print('=> Model')
print(model)

print('=> Criterion')
print(criterion)

-- 3. Convert model to CUDA
print('==> Converting model to CUDA')
-- model is converted to CUDA in the init script itself
-- model = model:cuda()
criterion:cuda()

collectgarbage()