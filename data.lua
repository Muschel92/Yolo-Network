image_roidb_train = {}
--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local ffi = require 'ffi'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

-- This script contains the logic to create K threads for parallel data-loading.
-- For the data-loading details, look at donkey.lua
-------------------------------------------------------------------------------
print('==> Loading Training Data')

if opt.prepare_image_roidb then
  train_roidb = torch.load(opt.data_path .. 'train_roidb.t7')
  
  image_roidb_train = prepare_image_roidb(opt, train_roidb)
  
  torch.save( opt.data_path .. 'train_roidb_all.t7', image_roidb_train)
  
  if opt.do_validation then
    val_roidb = torch.load(opt.data_path .. 'val_roidb.t7')
    image_roidb_val = prepare_image_roidb(opt, val_roidb)
    torch.save( opt.data_path .. 'val_roidb_all.t7', image_roidb_val)
  end
  
  
else
  image_roidb_train = torch.load(opt.data_path .. 'train_roidb_all.t7')
  
  if opt.do_validation then
     image_roidb_val = torch.load(opt.data_path .. 'val_roidb_all.t7')
  end
end

image_mean = torch.load( opt.data_path ..'meanImage.t7')
image_mean:div(255)


--[[
do -- start K datathreads (donkeys)
   if opt.nDonkeys > 0 then
      local options = opt -- make an upvalue to serialize over to donkey threads
      donkeys = Threads(
         opt.nDonkeys,
         function()
            require 'torch'
         end,
         function(idx)
            opt = options -- pass to all donkeys via upvalue
            tid = idx
            local seed = opt.manualSeed + idx
            torch.manualSeed(seed)
            print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
            paths.dofile('donkey.lua')
         end
      );
   else -- single threaded data loading. useful for debugging
      paths.dofile('donkey.lua')
      donkeys = {}
      function donkeys:addjob(f1, f2) f2(f1()) end
      function donkeys:synchronize() end
   end
end
]]--
