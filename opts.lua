
--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('YOLO Network')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------

    cmd:option('-cache', '/data/ethierer/ObjectDetection/YOLO_Network/save/', 'subdirectory in which to save/log experiments')
    cmd:option('-manualSeed',         2, 'Manually set RNG seed')
    cmd:option('-GPU',                1, 'Default preferred GPU')
    cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
    cmd:option('-backend',     'cudnn', 'Options: cudnn | nn')
    ------------- Preprocessing options ---------------
    cmd:option('-prepare_image_roidb',   false, 'True if image roidbs need to be calculated')
    
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        2, 'number of donkeys to initialize (data loading threads)')
    cmd:option('-imageSize',       torch.Tensor{448,448},    'Smallest side of the resized image')
    cmd:option('-cropSize',        224,    'Height and Width of image crop to be used as input layer')
    cmd:option('-nClasses',        20, 'number of classes in the dataset')
    cmd:option('-data_path',        '/data/ethierer/ObjectDetection/YOLO_Network/databases/VOC2007/Data_For_Debug/', 'path to dataset')
    
    ------------- Training options --------------------
    cmd:option('-do_validation',         false,    'If validation should be calculated')
    cmd:option('-nEpochs',         135,    'Number of total epochs to run')
    cmd:option('-epochSize',       100, 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       8,   'mini-batch size (1 = pure stochastic)')
    
    ---------- Optimization options ----------------------
    cmd:option('-LR',    0.0, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     5e-4, 'weight decay')
    
    ---------- Loss Options -----------------------------------
    cmd:option('-reg_loss',     5, 'weight for regression loss')
    cmd:option('-conf_loss',     1, 'weight for confidence loss')
    cmd:option('-neg_loss',     0.5, 'weight for negative confidence loss')
    cmd:option('-class_loss',     1, 'weight for classification loss')
    
    ---------- Model options ----------------------------------
    cmd:option('-netType',     'res_yolo_type10', 'Options: alexnet | overfeat | alexnetowtbn | vgg | googlenet')
    cmd:option('-netPath',     '/data/ethierer/ObjectDetection/YOLO_Network/model/', 'Path to the Yolo model')
    cmd:option('-retrain',     'none', 'provide path to model to retrain with')
    cmd:option('-new_model',     false, 'if a new model should be created')
    
    ----------- Saving Options ----------------------------------
    cmd:option('-save_step',     5, 'if a new model should be created')
    
    ----------- Testing options ---------------------------------
    cmd:option('-test_thresh',  0.5, 'provide path to an optimState to reload from')
    
    cmd:text()
    
    -----------YOLO Options -------------------------------------
    cmd:option('-grid_size',     torch.Tensor{7,7}, 'number of grid cells in each direction of image')
    cmd:option('-boxes_per_grid',     2, 'number of boxes a grid cell predicts')
    
    
    local opt = cmd:parse(arg or {})
    -- add commandline specified options
    opt.save = paths.concat(opt.cache,
                            cmd:string(opt.netType, opt,
                                       {netType=true, retrain=true, optimState=true, cache=true, data=true}))
    -- add date/time
    opt.save = paths.concat(opt.save, '' .. os.date():gsub(' ',''):gsub(':','_'))
    
    print('Saving everything to: ' .. opt.save)
    os.execute('mkdir -p ' .. opt.save)
    
    local im_path = opt.save .. '/Images'
    local model_path = opt.save ..'/Models'
    
    paths.mkdir(im_path)
    paths.mkdir(model_path)
    
    opt.save = opt.save .. '/'

    return opt
end

return M